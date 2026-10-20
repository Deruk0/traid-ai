from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from traid_ai.dashboard import SessionManager, parse_settings
from traid_ai.engine import Engine
from traid_ai.models import Config, D, Quote, stamp, utc_now
from traid_ai.qwen import (MODEL, ModelConnection, ModelError, OllamaClient,
                           QwenStrategy, build_snapshot, validate_answer)
from traid_ai.store import Store
from test_alpaca import await_condition

DIGEST = "a" * 64


class ControlledModel:
    def __init__(self, action="REQUEST_ENTRY"):
        self.action = action
        self.release = threading.Event()
        self.release.set()
        self.called = threading.Event()
        self.calls = []
        self.fail = False
    def status(self):
        return dict(model=MODEL, digest=DIGEST, parameter_size="4.3B", quantization="Q4_K_M")
    def decide(self, snapshot, *, digest, timeout=20):
        self.calls.append(deepcopy(snapshot))
        self.called.set()
        self.release.wait(2)
        if self.fail:
            raise ModelError("offline")
        return dict(snapshot_id=snapshot["snapshot_id"], action=self.action,
                    reason="Тестовое намерение", latency_ms=10)


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/"paper.sqlite3")
        self.now = utc_now()
        self.config = Config(source="kraken-public", strategy_version="qwen-local-v1", llm_digest=DIGEST,decision_policy="legacy",exit_policy="fixed-stop-v1")
        self.engine = Engine.start(self.store, self.config, self.now, self.now+timedelta(hours=1))
        self.client = ControlledModel()
        self.strategy = QwenStrategy(client=self.client)
        self.engine.strategy = self.strategy
        self.engine.state.history = ["100"]*5
        self.store.commit_step(self.engine.state,"warm",stamp(self.now),[])
    def tearDown(self):
        self.client.release.set()
        self.strategy.close()
        self.store.close()
        self.tmp.cleanup()
    def tick(self, *, price="100", delay=1, **kw):
        self.now += timedelta(seconds=delay)
        q=Quote(stamp(self.now),stamp(self.now),D(price),D(price)+D(".01"),"kraken-public")
        return self.engine.step(self.now,q,**kw)
    def done(self):
        await_condition(lambda:self.strategy.job and self.strategy.job["done"])

    def test_valid_decision_uses_risk_and_saved_snapshot_without_keys(self):
        self.tick()
        self.done()
        self.assertEqual(self.engine.state.entries,0)
        self.tick()
        self.assertEqual(self.engine.state.entries,1)
        self.assertLessEqual(D(self.engine.state.quantity)*D(self.engine.state.entry_price)*D("1.001"),D("250"))
        snap=self.client.calls[0]
        self.assertEqual(snap["market"]["prior_sample_count"],5)
        self.assertNotIn("broker",snap)
        self.assertNotIn("account_id",json.dumps(snap))
        kinds=[e["kind"] for e in self.store.events(self.engine.state.session_id)]
        self.assertLess(kinds.index("AI_REQUEST"),kinds.index("AI_DECISION"))
        self.assertTrue(self.engine.state.ai["status"]=="READY")

    def test_slow_model_never_blocks_deadline_or_reenters_after_close(self):
        self.client.release.clear()
        started=time.monotonic()
        self.tick()
        self.assertLess(time.monotonic()-started,.5)
        self.tick(delay=3600)
        self.assertEqual(self.engine.state.status,"CLOSED")
        self.client.release.set()
        self.done()
        self.tick()
        self.assertEqual(self.engine.state.entries,0)

    def test_thinking_finalization_stays_in_background_and_cannot_outlive_deadline(self):
        client=OllamaClient()
        self.strategy.client=client
        self.client.release.clear()
        finalizing=threading.Event()
        calls=[]
        def request(path,payload,**kwargs):
            calls.append(payload)
            if payload["think"]:
                return dict(model=MODEL,done=True,done_reason="length",
                            message=dict(thinking="Unfinished entry hypothesis",content=""))
            finalizing.set()
            self.client.release.wait(2)
            snapshot=json.loads(payload["messages"][1]["content"])
            return dict(model=MODEL,done=True,done_reason="stop",message=dict(content=json.dumps(dict(
                snapshot_id=snapshot["snapshot_id"],action="REQUEST_ENTRY",reason="Проверка"))))
        with patch.object(client,"status",return_value=dict(digest=DIGEST)), \
                patch.object(client,"_request",side_effect=request):
            started=time.monotonic()
            self.tick()
            self.assertLess(time.monotonic()-started,.5)
            self.assertTrue(finalizing.wait(1))
            self.tick(delay=3600)
            self.assertEqual(self.engine.state.status,"CLOSED")
            self.client.release.set();self.done();self.tick()
        self.assertEqual(len(calls),2)
        self.assertEqual(self.engine.state.entries,0)

    def test_next_request_receives_calculations_changes_and_confirmed_fill(self):
        self.tick(); self.done(); self.tick()
        self.tick(delay=10, price="100.2"); self.done()
        snapshot = self.client.calls[1]
        self.assertEqual(snapshot["analytics"]["sample_count"], 3)
        self.assertEqual(snapshot["analytics"]["windows"][0]["status"], "WARMUP")
        self.assertGreater(D(snapshot["changes"]["mid_change_pct"]), 0)
        self.assertEqual(snapshot["memory"]["entries"], 1)
        self.assertEqual(snapshot["memory"]["last_decision"]["action"], "REQUEST_ENTRY")
        self.assertTrue(snapshot["memory"]["last_decision"]["accepted"])
        self.assertEqual(snapshot["memory"]["recent_events"][-1]["kind"], "PAPER_FILL")
        self.assertEqual(snapshot["memory"]["recent_events"][-1]["side"], "BUY")
        self.assertEqual([o["action"] for o in snapshot["options"]], ["WAIT", "REQUEST_EXIT"])
        self.assertEqual(self.engine.state.ai["context_summary"]["memory_event_count"], 1)

    def test_stale_answer_and_position_change_are_rejected(self):
        self.tick(); self.done(); self.tick(delay=31)
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["status"],"REJECTED")
        for _ in range(5):self.tick()
        self.done()
        self.engine.state.cash="999"
        self.store.commit_step(self.engine.state,"changed",stamp(self.now),[])
        self.tick()
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["status"],"REJECTED")

    def test_paused_request_cannot_be_reused_after_continue(self):
        self.client.release.clear()
        self.tick()
        self.tick(pause_entries=True)
        self.client.release.set(); self.done()
        self.tick(pause_entries=False)
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["status"],"REJECTED")

    def test_model_failure_waits_and_protection_exits_without_new_inference(self):
        self.client.fail=True
        self.tick(); self.done(); self.tick()
        self.assertEqual(self.engine.state.entries,0)
        self.client.fail=False
        self.tick(delay=10); self.done(); self.tick()
        self.assertEqual(self.engine.state.entries,1)
        count=len(self.client.calls)
        self.tick(price="90")
        self.assertEqual(self.engine.state.quantity,"0")
        self.assertEqual(len(self.client.calls),count)

    def test_unpersisted_request_cannot_authorize_entry(self):
        original=self.store.commit_step
        def fail(state,step_id,now,events):
            if any(k=="AI_REQUEST" for k,p in events):
                raise OSError("disk")
            return original(state,step_id,now,events)
        self.store.commit_step=fail
        with self.assertRaises(OSError):self.tick()
        self.done()
        self.store.commit_step=original
        self.tick()
        self.assertEqual(self.engine.state.entries,0)

    def test_schema_rejects_wrong_snapshot_extra_fields_and_wrong_action(self):
        snap={"snapshot_id":"test","allowed_actions":["WAIT"]}
        base=dict(snapshot_id="test",action="WAIT",reason="Нет данных")
        for invalid in [dict(base,snapshot_id="other"),dict(base,action="REQUEST_ENTRY"),
                        dict(base,quantity="100"),dict(base,reason=""),dict(base,reason=5)]:
            with self.assertRaises(ModelError):validate_answer(invalid,snap)

    def test_loopback_schema_request_and_digest_change_blocking(self):
        client=OllamaClient()
        snap={"snapshot_id":"test","allowed_actions":["WAIT"]}
        row=dict(name=MODEL,digest=DIGEST,details=dict(family="qwen35",parameter_size="4.3B"))
        reply=dict(model=MODEL,done=True,message=dict(content=json.dumps(dict(
            snapshot_id="test",action="WAIT",reason="Нет данных"))))
        responses=[{"models":[row]},reply]
        calls=[]
        def opened(req,**kwargs):
            calls.append(req)
            return io.BytesIO(json.dumps(responses.pop(0)).encode())
        with patch.object(client.opener,"open",side_effect=opened):
            client.decide(snap,digest=DIGEST)
        self.assertTrue(all(r.full_url.startswith("http://127.0.0.1:11434/") for r in calls))
        payload=json.loads(calls[1].data)
        self.assertFalse(payload["think"])
        self.assertNotIn("tools",payload)
        self.assertEqual(payload["format"]["properties"]["snapshot_id"]["const"],"test")
        with patch.object(client,"status",return_value=dict(digest="b"*64)):
            with self.assertRaises(ModelError):client.decide(snap,digest=DIGEST)

    def test_old_config_and_state_remain_readable_and_qwen_start_requires_connection(self):
        old=self.config.to_dict();old.pop("llm_digest");old["strategy_version"]="demo-momentum-v1"
        self.assertEqual(Config.from_dict(old).llm_digest,"")
        with self.assertRaises(ValueError):parse_settings(dict(mode="demo",strategy="qwen"))
        manager=SessionManager(Path(self.tmp.name)/"separate.sqlite3")
        with self.assertRaises(ValueError):manager.launch(payload=dict(mode="paper",strategy="qwen"))
        self.assertFalse(manager.running)

    def test_connection_probe_does_not_trade_and_pins_model(self):
        connection=ModelConnection(client_factory=lambda model=MODEL:ControlledModel("WAIT"))
        connection.connect()
        await_condition(lambda:not connection.view()["checking"])
        self.assertTrue(connection.view()["connected"])
        self.assertEqual(connection.view()["info"]["digest"],DIGEST)

    def test_qwen_intents_flow_through_paper_executor_without_direct_tools(self):
        from traid_ai.broker import BrokerEngine
        from test_broker import FakeBroker
        client=FakeBroker();client.auto_fill=True
        cfg=Config(source="alpaca-paper",strategy_version="qwen-local-v1",llm_digest=DIGEST,decision_policy="legacy",exit_policy="fixed-stop-v1")
        engine=BrokerEngine.start_paper(self.store,cfg,self.now,self.now+timedelta(hours=1),client)
        engine.strategy=self.strategy
        engine.state.history=["100"]*5
        self.store.commit_step(engine.state,"broker-warm",stamp(self.now),[])
        def tick(delay=1,**kwargs):
            self.now+=timedelta(seconds=delay)
            return engine.step(self.now,Quote(stamp(self.now),stamp(self.now),D("99.99"),D("100"),"alpaca-paper"),**kwargs)
        tick();self.done();tick();tick()
        self.assertEqual([p["side"] for p in client.posts],["buy"])
        self.client.action="REQUEST_EXIT"
        tick(10);self.done();tick();tick(stop=True)
        self.assertTrue(any(e["kind"] == "BROKER_FILL" for e in self.client.calls[1]["memory"]["recent_events"]))
        self.assertEqual([p["side"] for p in client.posts],["buy","sell"])
        self.assertEqual(engine.state.status,"CLOSED")
        self.assertEqual(client.qty,0)


if __name__=="__main__":unittest.main()
