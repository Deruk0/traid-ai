from datetime import timedelta
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from traid_ai.alpaca import PaperClient, AlpacaError
from traid_ai.dashboard import SessionManager, DashboardServer, decision_flow
from traid_ai.engine import Engine
from traid_ai.models import Config, D, Quote, stamp, utc_now
from traid_ai.runtime import QuoteFeed
from traid_ai.store import Store
from test_alpaca import await_condition
from test_evaluation import dataset


class RuntimeTests(unittest.TestCase):
    def test_model_completion_wakes_dashboard_and_executes_without_another_quote(self):
        from test_qwen import ControlledModel
        class SingleQuoteFeed:
            def __init__(self,*args,**kwargs): self.taken=False
            def refresh(self): return False
            def take(self):
                if self.taken: return None,None
                self.taken=True
                now=stamp(utc_now())
                return Quote("single",now,D("100"),D("100.01"),"kraken-public"),None
            def view(self): return dict(status="READY")
            def close(self): pass
        class NoCandles:
            def __init__(self,*args): pass
            def refresh(self,now): return False
            def view(self,now): return dict(rows=[],summary={})
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"handoff.sqlite3"
            store=Store(path);now=utc_now()
            config=Config(source="kraken-public",strategy_version="qwen-local-v1",
                          llm_digest="a"*64,decision_policy="legacy")
            engine=Engine.start(store,config,now,now+timedelta(minutes=2))
            engine.state.history=["100"]*5
            store.commit_step(engine.state,"warmup",stamp(now),[])
            identity=engine.state.session_id
            engine.strategy.close();store.close()
            client=ControlledModel();client.release.clear()
            manager=SessionManager(path)
            with patch("traid_ai.dashboard.QuoteFeed",SingleQuoteFeed), \
                    patch("traid_ai.dashboard.CandleFeed",NoCandles), \
                    patch("traid_ai.qwen.OllamaClient",return_value=client):
                try:
                    manager.launch(session_id=identity)
                    self.assertTrue(client.called.wait(1))
                    started=time.monotonic()
                    client.release.set()
                    await_condition(lambda:manager.view(identity)["state"]["entries"]==1)
                    self.assertLess(time.monotonic()-started,2)
                    view=manager.view(identity,full_events=True)
                    self.assertEqual(sum(e["kind"]=="SNAPSHOT" for e in view["events"]),1)
                    self.assertEqual(sum(e["kind"]=="QUOTE_RECHECK" for e in view["events"]),1)
                finally:
                    client.release.set();manager.shutdown()

    def test_quote_loading_does_not_block_and_close_discards_late_result(self):
        release=threading.Event()
        feed=QuoteFeed(lambda:release.wait(2),interval_seconds=10)
        start=time.monotonic()
        self.assertTrue(feed.refresh())
        self.assertEqual(feed.take(),(None,None))
        self.assertLess(time.monotonic()-start,.5)
        self.assertFalse(feed.refresh())
        feed.close();release.set()
        self.assertEqual(feed.take(),(None,None))

    def test_quote_requests_cannot_exceed_ten_second_rate(self):
        clock=[0]
        feed=QuoteFeed(lambda:"quote",clock=lambda:clock[0])
        feed.refresh()
        await_condition(lambda:not feed._running)
        self.assertEqual(feed.take(),("quote",None))
        clock[0]=9
        self.assertFalse(feed.refresh())
        clock[0]=10
        self.assertTrue(feed.refresh())
        feed.close()

    def test_watchdog_stops_owner_without_waiting_for_quote_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            release=threading.Event()
            manager=SessionManager(Path(tmp)/"state.sqlite3")
            def blocked():
                release.wait(5)
                return None
            with patch("traid_ai.dashboard.public_quote",side_effect=blocked):
                manager.launch(payload=dict(mode="paper",duration_minutes="2"))
                manager._watchdog_trip("SYSTEM_RESUME_GAP")
                await_condition(lambda:not manager.running)
                view=manager.view()
                self.assertEqual(view["report"]["status"],"CLOSED")
                self.assertEqual(view["report"]["exit_reason"],"SYSTEM_RESUME_GAP")
                self.assertTrue(any(e["kind"]=="WATCHDOG_TRIPPED" for e in view["events"]))
                self.assertEqual(view["report"]["entries"],0)
                release.set()
            manager.shutdown()

    def test_worker_failure_persists_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=SessionManager(Path(tmp)/"state.sqlite3")
            with patch.object(Engine,"step",side_effect=RuntimeError("failed")):
                try:
                    manager.launch(payload=dict(mode="demo"))
                except ValueError:
                    pass
                await_condition(lambda:not manager.running)
                self.assertEqual(manager.view()["report"]["status"],"UNRESOLVED")
                self.assertEqual(manager.view()["report"]["exit_reason"],"WORKER_ERROR")
            manager.shutdown()

    def test_explicit_fee_reconciliation_is_repeatable_and_persists_actual_fees(self):
        from test_accounting import AccountingTests
        fixture=AccountingTests();fixture.setUp()
        class Client:
            def account(self):return fixture.account
            def activities(self,after,until):return fixture.activities
            def positions(self):return []
            def open_orders(self):return []
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.sqlite3"
            store=Store(path);store.create(fixture.config,fixture.state)
            for i,event in enumerate(fixture.events):
                store.commit_step(fixture.state,str(i),event["occurred_at"],[(event["kind"],event["payload"])])
            store.close()
            manager=SessionManager(path)
            with patch.object(manager.alpaca,"execution_client",return_value=Client()):
                for _ in range(2):
                    manager.analyze_session(fixture.state.session_id,"accounting")
                    await_condition(lambda:not manager.maintenance["running"])
                    self.assertIsNone(manager.maintenance["error"])
                    report=manager.view(fixture.state.session_id)["report"]
                    self.assertTrue(report["pnl_is_final"])
                    self.assertEqual(D(report["fees"]),D("1.5"))
                    self.assertEqual(D(report["net_pnl_at_last_quote"]),D("8.4"))
            manager.shutdown()

    def test_late_fee_network_read_cannot_block_execution_owner(self):
        from test_broker import FakeBroker
        from traid_ai.broker import BrokerEngine
        release=threading.Event()
        client=FakeBroker()
        client.activities=lambda after,until:release.wait(2)
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/"state.sqlite3")
            now=utc_now()
            engine=BrokerEngine.start_paper(store,Config(source="alpaca-paper"),now,now+timedelta(hours=1),client)
            start=time.monotonic()
            self.assertFalse(engine._late_fees(now,client.account(),[],[]))
            self.assertLess(time.monotonic()-start,.5)
            release.set();store.close()

    def test_history_provider_is_saved_and_visible_to_model(self):
        from traid_ai.qwen import build_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/"state.sqlite3")
            now=utc_now().replace(second=0,microsecond=0)
            cfg=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
            engine=Engine.start(store,cfg,now,now+timedelta(hours=1))
            from traid_ai.history import summarize_candles
            rows=[dict(source="kraken-public",instrument="BTC/USD",start_at=stamp(now-timedelta(minutes=2)),
                       end_at=stamp(now-timedelta(minutes=1)),available_at=stamp(now),
                       open="100",high="101",low="99",close="100",volume="2")]
            context=dict(rows=rows,summary=summarize_candles(rows,"kraken-public",now))
            engine.candle_provider=lambda when:context
            quote=Quote("q",stamp(now),D("100"),D("101"),"kraken-public")
            engine.step(now,quote)
            snapshot=build_snapshot(cfg,store.load(engine.state.session_id)[1],quote,now)
            self.assertEqual(snapshot["candles"]["count"],1)
            self.assertTrue(snapshot["context_summary"]["volume_available"])
            self.assertTrue(any(e["kind"]=="CANDLE_HISTORY" for e in store.events(engine.state.session_id)))
            engine.strategy.close();store.close()

    def test_flow_does_not_confuse_approval_and_partial_execution(self):
        def event(kind,payload):
            return dict(kind=kind,payload=payload,occurred_at=stamp(utc_now()))
        events=[event("DECISION",dict(decision_id="d",action="REQUEST_ENTRY",reason="TEST")),
                event("RISK_VERDICT",dict(decision_id="d",approved=True,reason="WITHIN_LIMITS")),
                event("ORDER_PREPARED",dict(decision_id="d",client_order_id="c"))]
        self.assertEqual(decision_flow(events)[0]["execution"],"PREPARED")
        events.append(event("BROKER_FILL",dict(decision_id="d",quantity=".1")))
        row=decision_flow(events)[0]
        self.assertEqual(row["execution"],"PARTIALLY_FILLED")
        self.assertEqual(row["filled_quantity"],"0.1")
        events.append(event("RISK_VERDICT",dict(decision_id="d",approved=False,reason="BROKER_NOT_READY")))
        self.assertFalse(decision_flow(events)[0]["risk"]["approved"])

    def test_old_qwen_events_keep_declared_origin_without_inventing_snapshot_links(self):
        row=decision_flow([dict(kind="DECISION",occurred_at=stamp(utc_now()),payload=dict(
            decision_id="legacy",action="WAIT",reason="QWEN_DECISION",strategy_version="qwen-local-v1"))])[0]
        self.assertEqual(row["origin"],"QWEN")
        self.assertEqual(row["execution"],"NONE")

    def test_comparison_http_requires_token_and_persists_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.sqlite3"
            export=dataset()
            store=Store(path)
            from traid_ai.models import State
            cfg=Config.from_dict(export["config"])
            state=State(**export["state"])
            state.status="CLOSED"
            store.create(cfg,state)
            for i,event in enumerate(export["events"]):
                store.commit_step(state,str(i),event["occurred_at"],[(event["kind"],event["payload"])])
            store.close()
            manager=SessionManager(path)
            server=DashboardServer(("127.0.0.1",0),manager)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                conn=http.client.HTTPConnection("127.0.0.1",server.server_port)
                body=json.dumps(dict(session_id=state.session_id))
                conn.request("POST","/api/compare",body,{"Content-Type":"application/json"})
                reply=conn.getresponse();reply.read();self.assertEqual(reply.status,403)
                conn.request("POST","/api/compare",body,{"Content-Type":"application/json","X-Control-Token":server.token})
                reply=conn.getresponse();reply.read();self.assertEqual(reply.status,200)
                await_condition(lambda:not manager.maintenance["running"])
                self.assertIsNone(manager.maintenance["error"])
                self.assertIn("dataset_sha256",manager.view(state.session_id)["comparison"])
                conn.close()
            finally:
                manager.shutdown();server.shutdown();server.server_close();thread.join()


class NewAlpacaReadsTests(unittest.TestCase):
    def test_activity_pagination_dedup_cursor_bound_and_redaction(self):
        client=PaperClient("key","secret")
        page=[dict(id=str(i),activity_type="FEE",date="2026-10-14",net_amount="-.01",secret="secret") for i in range(100)]
        with patch.object(client,"_get",side_effect=[page,[]]) as get:
            rows=client.activities("2026-10-13T00:00:00Z","2026-10-14T23:00:00Z")
        self.assertEqual(len(rows),100)
        self.assertNotIn("secret",json.dumps(rows))
        self.assertIn("page_token=99",get.call_args.args[0])
        with patch.object(client,"_get",side_effect=[page,page]):
            with self.assertRaises(AlpacaError):
                client.activities("2026-10-13T00:00:00Z","2026-10-14T23:00:00Z")

    def test_history_uses_separate_client_and_only_data_get(self):
        client=PaperClient("key","secret")
        data=client.data_client()
        self.assertIsNot(client._io_lock,data._io_lock)
        with patch.object(data,"_get",return_value=dict(bars={"BTC/USD":[]},next_page_token=None)) as get:
            self.assertEqual(data.candles(utc_now()),[])
        self.assertTrue(get.call_args.kwargs["market_data"])
        self.assertTrue(get.call_args.args[0].startswith("/v1beta3/crypto/us/bars?"))


if __name__=="__main__":unittest.main()
