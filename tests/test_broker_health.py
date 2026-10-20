from datetime import timedelta
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError

from traid_ai.alpaca import PaperClient, AlpacaError
from traid_ai.broker import BrokerEngine
from traid_ai.broker_io import BrokerReadCycle
from traid_ai.dashboard import SessionManager
from traid_ai.engine import Engine
from traid_ai.health import technical_health
from traid_ai.models import Config, D, Quote, stamp, utc_now
from traid_ai.store import Store
from test_alpaca import await_condition, ACCOUNT, Reply
from test_broker import FakeBroker, EntryRule
import json
import http.client
from traid_ai.dashboard import DashboardServer


class ReadCycleTests(unittest.TestCase):
    def test_slow_read_is_nonblocking_and_single_flight(self):
        client = FakeBroker()
        release, entered = threading.Event(), threading.Event()
        calls = []
        def slow():
            calls.append("account"); entered.set(); release.wait(3)
            return FakeBroker.account(client)
        client.account = slow
        cycle = BrokerReadCycle(client)
        try:
            start = time.monotonic()
            self.assertIsNone(cycle.poll())
            self.assertTrue(entered.wait(.5))
            for _ in range(3): self.assertIsNone(cycle.poll())
            self.assertLess(time.monotonic()-start, .5)
            self.assertEqual(calls, ["account"])
            self.assertEqual(cycle.view()["stage"], "account")
            release.set()
            await_condition(lambda:cycle.job["done"].is_set())
            self.assertEqual(cycle.poll()["account"]["account_id"], client.account_id)
        finally:
            cycle.close(); release.set()

    def test_timeout_discards_late_data_without_overlapping_reads(self):
        client, release = FakeBroker(), threading.Event()
        clock = [0]
        calls = []
        def slow():
            calls.append("account"); release.wait(3); return FakeBroker.account(client)
        client.account = slow
        cycle = BrokerReadCycle(client, clock=lambda:clock[0])
        try:
            self.assertIsNone(cycle.poll())
            clock[0] = 6
            for _ in range(2):
                with self.assertRaises(AlpacaError):cycle.poll()
            self.assertEqual(calls, ["account"])
            self.assertEqual(cycle.view()["status"], "TIMED_OUT")
            release.set(); await_condition(lambda:cycle.job["done"].is_set())
            with self.assertRaisesRegex(AlpacaError,"Поздний"):cycle.poll()
            client.account = lambda:FakeBroker.account(client)
            self.assertIsNotNone(cycle.poll())
        finally:
            cycle.close(); release.set()

    def test_close_stops_followup_reads_and_never_has_write_authority(self):
        client, release = FakeBroker(), threading.Event()
        reads=[]
        client.account=lambda:(release.wait(3), FakeBroker.account(client))[1]
        client.positions=lambda:reads.append("positions") or []
        cycle=BrokerReadCycle(client)
        self.assertIsNone(cycle.poll())
        cycle.close(); release.set()
        await_condition(lambda:cycle.job["done"].is_set())
        self.assertEqual(reads,[])
        self.assertEqual(client.posts,[])
        with self.assertRaises(AlpacaError):cycle.poll()

    def test_budget_covers_all_requests_and_disables_read_retry(self):
        client=PaperClient("key","secret")
        with patch.object(client._opener,"open",side_effect=URLError("offline")) as opened:
            with client.request_budget(.5):
                with self.assertRaises(AlpacaError):client.account()
            self.assertEqual(opened.call_count,1)
            self.assertLessEqual(opened.call_args.kwargs["timeout"],.5)
        with patch("traid_ai.alpaca.time.monotonic",side_effect=[10,11]):
            with client.request_budget(.5):
                with patch.object(client._opener,"open") as opened:
                    with self.assertRaises(AlpacaError):client.account()
                    opened.assert_not_called()
        # Budget is thread local and is restored when leaving the scope.
        self.assertEqual(client._timeout(),10)


class OwnerAndHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/"test.sqlite3")
        self.now=utc_now()
        self.client=FakeBroker()
        self.engine=BrokerEngine.start_paper(self.store,Config(source="alpaca-paper"),self.now,
                                            self.now+timedelta(hours=1),self.client)
        self.engine.strategy=EntryRule()

    def tearDown(self):
        self.engine.close(); self.store.close(); self.tmp.cleanup()

    def quote(self):
        return Quote(stamp(self.now),stamp(self.now),D("99.99"),D("100"),"alpaca-paper")

    def deferred_entry(self):
        from traid_ai.strategy import Intent
        calls = []
        class OneShotEntry:
            version = "test-one-shot"
            def decide(self, history, mid, position):
                calls.append(mid)
                return Intent("REQUEST_ENTRY" if len(calls) == 1 else "WAIT", "TEST")
        self.engine.strategy = OneShotEntry()
        release = threading.Event()
        self.client.account = lambda: (release.wait(3), FakeBroker.account(self.client))[1]
        self.addCleanup(release.set)
        return release, calls

    def test_deferred_entry_survives_read_and_uses_quote_once_after_reconciliation(self):
        release, calls = self.deferred_entry()
        self.engine.step(self.now, self.quote())
        self.assertEqual(calls, [])
        self.assertEqual(self.client.posts, [])
        history = list(self.engine.state.history)
        release.set()
        await_condition(lambda: self.engine.read_cycle.job["done"].is_set())
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual(len(calls), 1)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy"])
        self.assertEqual(self.engine.state.history, history)
        self.assertEqual(sum(e["kind"] == "SNAPSHOT" for e in
                             self.store.events(self.engine.state.session_id)), 1)
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual(len(self.client.posts), 1)

    def test_deferred_entry_is_not_retried_with_expired_quote(self):
        release, calls = self.deferred_entry()
        self.engine.step(self.now, self.quote())
        release.set()
        await_condition(lambda: self.engine.read_cycle.job["done"].is_set())
        self.now += timedelta(seconds=self.engine.config.max_quote_age + 1)
        self.engine.step(self.now, None)
        self.assertEqual(calls, [])
        self.assertEqual(self.client.posts, [])

    def test_completed_qwen9b_entry_waits_for_broker_then_passes_original_snapshot_checks(self):
        from traid_ai.qwen import QwenStrategy
        class Model:
            def decide(self, snapshot, *, digest):
                return dict(snapshot_id=snapshot["snapshot_id"], action="REQUEST_ENTRY", reason="Test entry")
        self.engine.close()
        config=Config(source="alpaca-paper",strategy_version="qwen-local-v1",
                      decision_policy="legacy",llm_model="qwen3.5:9b",llm_digest="a"*64)
        self.engine=BrokerEngine.start_paper(self.store,config,self.now,
                                             self.now+timedelta(hours=1),self.client)
        self.engine.strategy.close()
        self.engine.strategy=QwenStrategy(client=Model(),model="qwen3.5:9b")
        self.addCleanup(self.engine.strategy.close)
        self.engine.state.history=["100"]*5
        self.store.commit_step(self.engine.state,"warmup",stamp(self.now),[])
        self.engine.step(self.now,self.quote())
        await_condition(lambda:self.engine.strategy.job and self.engine.strategy.job["done"])
        snapshot_id=self.engine.strategy.job["snapshot"]["snapshot_id"]
        release=threading.Event()
        self.addCleanup(release.set)
        self.client.account=lambda:(release.wait(3),FakeBroker.account(self.client))[1]
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,self.quote())
        self.assertIsNotNone(self.engine.strategy.job)
        self.assertEqual(self.client.posts,[])
        release.set()
        await_condition(lambda:self.engine.read_cycle.job["done"].is_set())
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,None)
        self.assertEqual([p["side"] for p in self.client.posts],["buy"])
        decisions=[e["payload"] for e in self.store.events(self.engine.state.session_id)
                   if e["kind"]=="AI_DECISION"]
        self.assertEqual(len(decisions),1)
        self.assertTrue(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["snapshot_id"],snapshot_id)

    def assert_deferred_control(self, **control):
        release, calls = self.deferred_entry()
        self.engine.step(self.now, self.quote(), **control)
        release.set()
        await_condition(lambda: self.engine.read_cycle.job["done"].is_set())
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual(self.client.posts, [])

    def test_stop_during_read_prevents_deferred_buy(self):
        self.assert_deferred_control(stop=True)

    def test_pause_during_read_prevents_deferred_buy(self):
        self.assert_deferred_control(pause_entries=True)

    def test_blocked_read_accepts_quote_and_stop_without_submitting(self):
        release=threading.Event()
        self.client.account=lambda:(release.wait(3),FakeBroker.account(self.client))[1]
        try:
            started=time.monotonic()
            self.engine.step(self.now,self.quote())
            self.assertLess(time.monotonic()-started,.5)
            self.assertIsNotNone(self.engine.state.quote())
            self.assertFalse(self.engine.state.broker["reconciled"])
            self.now+=timedelta(seconds=1)
            self.engine.step(self.now,self.quote(),stop=True)
            self.assertEqual(self.engine.state.status,"UNRESOLVED")
            self.assertEqual(self.client.posts,[])
            self.assertEqual(self.engine.state.exit_reason,"USER_STOP")
        finally:release.set()

    def test_timeout_then_recovery_sends_only_protective_exit(self):
        self.client.auto_fill=True
        self.engine.step(self.now,self.quote())
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,self.quote())
        self.assertGreater(D(self.engine.state.quantity),0)
        release=threading.Event()
        self.client.account=lambda:(release.wait(3),FakeBroker.account(self.client))[1]
        clock=[0]; self.engine.read_cycle.clock=lambda:clock[0]
        try:
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote())
            clock[0]=6
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote())
            self.assertEqual(self.engine.state.status,"UNRESOLVED")
            self.assertEqual(len(self.client.posts),1)
            release.set(); await_condition(lambda:self.engine.read_cycle.job["done"].is_set())
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote()) # discard late
            self.client.account=lambda:FakeBroker.account(self.client)
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote())
            self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote())
            self.assertEqual(self.engine.state.status,"CLOSED")
            self.assertEqual(D(self.engine.state.quantity),0)
        finally:release.set()

    def test_health_reports_stale_quote_pending_order_and_read_stage(self):
        self.engine.step(self.now,self.quote())
        cfg,state=self.store.load(self.engine.state.session_id)
        result=technical_health(cfg,state,[],self.now+timedelta(seconds=40),active=True,
            broker_io=dict(status="READING",stage="positions",elapsed_seconds=2,budget_seconds=5))
        codes={b["code"] for b in result["blockers"]}
        self.assertTrue({"QUOTE_STALE","BROKER_UNCONFIRMED","ORDER_PENDING"}<=codes)
        self.assertEqual(next(c for c in result["components"] if c["component"]=="broker")["io"]["stage"],"positions")

    def test_unknown_submit_and_slow_reconcile_cannot_repeat_buy(self):
        self.client.auto_fill=True
        self.client.timeout_after_submit=True
        self.engine.step(self.now,self.quote())
        self.assertEqual(self.engine.state.broker["pending"]["phase"],"UNKNOWN")
        release=threading.Event()
        self.client.account=lambda:(release.wait(3),FakeBroker.account(self.client))[1]
        try:
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote(),stop=True)
            self.assertEqual(len(self.client.posts),1)
            self.assertIsNotNone(self.engine.state.broker["pending"])
            release.set(); await_condition(lambda:self.engine.read_cycle.job["done"].is_set())
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote(),stop=True)
            self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])
            self.now+=timedelta(seconds=1); self.engine.step(self.now,self.quote(),stop=True)
            self.assertEqual(self.engine.state.status,"CLOSED")
        finally:release.set()

    def test_archives_use_session_time_and_cannot_show_other_runtime(self):
        self.engine.step(self.now,self.quote())
        cfg,state=self.store.load(self.engine.state.session_id)
        result=technical_health(cfg,state,[],self.now+timedelta(days=10),active=False,
            broker_io=dict(status="TIMED_OUT",stage="foreign-session"))
        self.assertEqual(result["scope"],"ARCHIVE")
        rows={c["component"]:c for c in result["components"]}
        self.assertEqual(rows["quotes"]["status"],"READY")
        self.assertNotEqual(rows["broker"]["io"].get("stage"),"foreign-session")
        self.assertEqual(result["blockers"][0]["code"],"NOT_RUNNING")

    def test_qwen_latency_wait_and_data_feed_error_visible(self):
        from dataclasses import replace
        cfg=replace(self.engine.config,strategy_version="qwen-local-v1",llm_digest="a"*64)
        state=self.engine.state
        state.ai=dict(status="PENDING",processed_at=stamp(self.now-timedelta(seconds=7)),latency_ms=2500)
        result=technical_health(cfg,state,[],self.now,active=True,
                                 quote_io=dict(status="ERROR",error="Источник котировок недоступен"))
        rows={c["component"]:c for c in result["components"]}
        self.assertEqual(rows["qwen"]["waiting_seconds"],7)
        self.assertEqual(rows["qwen"]["latency_ms"],2500)
        self.assertEqual(rows["quotes"]["io"]["status"],"ERROR")

    def test_dashboard_remains_readable_and_stoppable_during_broker_read(self):
        # Use a separate empty DB so the manager owns every write.
        path=Path(self.tmp.name)/"manager.sqlite3"
        manager=SessionManager(path)
        client=FakeBroker(); release=threading.Event(); entered=threading.Event()
        calls=[0]
        def account():
            calls[0]+=1
            if calls[0]>=3:
                entered.set();release.wait(3)
            return FakeBroker.account(client)
        client.account=account
        server=DashboardServer(("127.0.0.1",0),manager)
        server_thread=threading.Thread(target=server.serve_forever,daemon=True)
        server_thread.start()
        def request(method,path,payload=None):
            connection=http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=2)
            connection.request(method,path,body=json.dumps(payload) if payload else None,
                headers={"Content-Type":"application/json","X-Control-Token":server.token})
            response=connection.getresponse(); body=response.read(); connection.close()
            self.assertEqual(response.status,200,body)
            return body
        try:
            with patch.object(manager.alpaca,"execution_client",return_value=client):
                session=manager.launch(payload=dict(mode="alpaca",duration_minutes="5"))
                self.assertTrue(entered.wait(1))
                started=time.monotonic(); view=json.loads(request("GET","/api/state?session="+session))
                self.assertLess(time.monotonic()-started,.5)
                self.assertEqual(view["health"]["scope"],"LIVE")
                broker=next(c for c in view["health"]["components"] if c["component"]=="broker")
                self.assertEqual(broker["io"]["stage"],"account")
                self.assertIn(b"health-grid",request("GET","/health.css"))
                request("POST","/api/control",dict(session_id=session,action="stop"))
                await_condition(lambda:not manager.running)
                self.assertEqual(manager.view(session)["report"]["status"],"UNRESOLVED")
                self.assertEqual(client.posts,[])
        finally:
            release.set(); manager.shutdown(); server.shutdown(); server.server_close(); server_thread.join()
