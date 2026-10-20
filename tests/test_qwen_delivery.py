"""Ready-result handoff, safe diagnostics and whole decision alternatives."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

from traid_ai.engine import Engine
from traid_ai.evaluation import compare_session
from traid_ai.models import Config, D, Quote, State, stamp, utc_now
from traid_ai.qwen import (ERROR_MESSAGES, ModelError, OllamaClient, QwenStrategy,
                           build_snapshot, schema, validate_answer)
from traid_ai.store import Store
from test_alpaca import await_condition
from test_qwen import ControlledModel
from test_broker import FakeBroker
from traid_ai.broker import BrokerEngine


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/"delivery.sqlite3")
        self.now=utc_now()
        config=Config(source="kraken-public",strategy_version="qwen-local-v1",
                      llm_model="qwen3.5:9b",llm_digest="a"*64,decision_policy="legacy")
        self.engine=Engine.start(self.store,config,self.now,self.now+timedelta(hours=1))
        self.engine.strategy.close()
        self.wake=threading.Event()
        self.client=ControlledModel()
        self.engine.strategy=QwenStrategy(client=self.client,model=config.llm_model,wake=self.wake)
        self.engine.state.history=["100"]*5
        self.store.commit_step(self.engine.state,"warmup",stamp(self.now),[])

    def tearDown(self):
        self.client.release.set()
        self.engine.strategy.close()
        self.store.close()
        self.tmp.cleanup()

    def request(self):
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,Quote("origin",stamp(self.now),D("100"),D("100.01"),"kraken-public"))
        await_condition(self.engine.strategy.has_ready_result)
        self.assertTrue(self.wake.wait(.5))
        return self.engine.strategy.job["snapshot"]["snapshot_id"]

    def test_ready_entry_runs_without_next_quote_and_replays_without_duplicate_observation(self):
        snapshot_id=self.request()
        samples=deepcopy(self.engine.state.quote_samples)
        history=list(self.engine.state.history)
        self.now+=timedelta(seconds=21)
        events=self.engine.step(self.now,None)
        self.assertEqual(self.engine.state.entries,1)
        self.assertEqual(self.engine.state.history,history)
        self.assertEqual(self.engine.state.quote_samples,samples)
        self.assertEqual(sum(k=="QUOTE_RECHECK" for k,p in events),1)
        self.assertFalse(any(k=="SNAPSHOT" for k,p in events))
        answer=next(p for k,p in events if k=="AI_DECISION")
        self.assertTrue(answer["accepted"])
        self.assertEqual(answer["snapshot_id"],snapshot_id)
        self.assertIsNone(answer["error_code"])
        self.assertEqual(answer["decision_age_ms"],21000)
        self.assertIsNotNone(answer["result_wait_ms"])
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,None)
        self.assertEqual(self.engine.state.entries,1)
        export=dict(config=self.engine.config.to_dict(),state=self.engine.state.to_dict(),
                    events=self.store.events(self.engine.state.session_id))
        report=compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"],1)

    def test_expired_original_snapshot_is_rejected_even_with_new_fresh_quote(self):
        self.request()
        self.now+=timedelta(seconds=31)
        self.engine.step(self.now,None)
        self.assertEqual(self.engine.state.entries,0)
        self.assertTrue(self.engine.strategy.has_ready_result())
        self.now+=timedelta(seconds=1)
        events=self.engine.step(self.now,Quote("new",stamp(self.now),D("100"),D("100.01"),"kraken-public"))
        answer=next(p for k,p in events if k=="AI_DECISION")
        self.assertFalse(answer["accepted"])
        self.assertEqual(answer["error_code"],"SNAPSHOT_EXPIRED")
        self.assertEqual(self.engine.state.entries,0)

    def test_pause_precedes_ready_entry(self):
        self.request()
        self.now+=timedelta(seconds=1)
        events=self.engine.step(self.now,None,pause_entries=True)
        self.assertEqual(self.engine.state.entries,0)
        answer=next(p for k,p in events if k=="AI_DECISION")
        self.assertEqual(answer["error_code"],"PORTFOLIO_CHANGED")

    def test_stop_precedes_ready_entry(self):
        self.request()
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,None,stop=True)
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.status,"CLOSED")

    def test_deadline_precedes_ready_entry(self):
        self.request()
        self.now+=timedelta(hours=1)
        self.engine.step(self.now,None)
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.status,"CLOSED")

    def test_broker_consumes_ready_result_without_prior_deferred_strategy(self):
        config=replace(self.engine.config,source="alpaca-paper")
        self.engine.strategy.close()
        broker=FakeBroker()
        self.engine=BrokerEngine.start_paper(self.store,config,self.now,
                                             self.now+timedelta(hours=1),broker)
        self.addCleanup(self.engine.close)
        self.engine.strategy.close()
        self.engine.strategy=QwenStrategy(client=self.client,model=config.llm_model,wake=self.wake)
        self.engine.state.history=["100"]*5
        self.store.commit_step(self.engine.state,"broker-warmup",stamp(self.now),[])
        self.engine.step(self.now,Quote("broker-quote",stamp(self.now),D("100"),D("100.01"),"alpaca-paper"))
        await_condition(self.engine.strategy.has_ready_result)
        self.assertFalse(self.engine._strategy_deferred)
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,None)
        self.assertEqual([p["side"] for p in broker.posts],["buy"])

    def test_diagnostic_code_reaches_state_and_journal_without_exception_text(self):
        def fail(*args,**kwargs):
            raise ModelError("private-secret-and-draft",code="MODEL_TIMEOUT")
        self.client.decide=fail
        self.request()
        self.now+=timedelta(seconds=1)
        self.engine.step(self.now,None)
        self.assertEqual(self.engine.state.ai["error_code"],"MODEL_TIMEOUT")
        self.assertEqual(self.engine.state.entries,0)
        events=self.store.events(self.engine.state.session_id)
        self.assertNotIn("private-secret",json.dumps(events))
        self.assertEqual(next(e["payload"]["error_code"] for e in events if e["kind"]=="AI_DECISION"),"MODEL_TIMEOUT")


class DiagnosticsAndSchemaTests(unittest.TestCase):
    def test_known_and_unexpected_exceptions_keep_codes_without_raw_exception_text(self):
        for exc,code in ((ModelError("secret draft",code="MODEL_TIMEOUT"),"MODEL_TIMEOUT"),
                         (ModelError("secret draft",code="CONTRACT_INVALID"),"CONTRACT_INVALID"),
                         (ModelError("secret draft",code="untrusted-secret"),"MODEL_ERROR"),
                         (RuntimeError("secret draft"),"INTERNAL_ERROR")):
            with self.subTest(code=code):
                wake=threading.Event()
                strategy=QwenStrategy(wake=wake)
                job=dict(snapshot={})
                strategy.job=job
                with patch.object(strategy.client,"decide",side_effect=exc):
                    strategy._run(job,"a"*64)
                self.assertTrue(wake.is_set())
                self.assertEqual(job["error_code"],code)
                self.assertEqual(job["error"],ERROR_MESSAGES[code])
                self.assertNotIn("secret",json.dumps(job))
                self.assertIsNotNone(job["elapsed_ms"])
                strategy.close()

    def test_closed_worker_cannot_publish_or_wake(self):
        wake=threading.Event()
        strategy=QwenStrategy(wake=wake)
        job=dict(snapshot={})
        strategy.job=job
        strategy.close()
        with patch.object(strategy.client,"decide",side_effect=RuntimeError("ignored")):
            strategy._run(job,"a"*64)
        self.assertFalse(wake.is_set())
        self.assertNotIn("done",job)

    def test_transport_classifies_timeout_offline_and_invalid_json(self):
        client=OllamaClient()
        for exc,code in ((TimeoutError(),"MODEL_TIMEOUT"),(URLError(TimeoutError()),"MODEL_TIMEOUT"),
                         (URLError("offline"),"MODEL_UNAVAILABLE"),(ValueError("secret"),"RESPONSE_INVALID_JSON")):
            with self.subTest(code=code),patch.object(client.opener,"open",side_effect=exc):
                with self.assertRaises(ModelError) as caught:client._request("/api/tags")
                self.assertEqual(caught.exception.code,code)

    def test_schema_expresses_only_complete_options_and_keeps_wire_contract(self):
        now=utc_now()
        config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
        state=State("schema",stamp(now),stamp(now+timedelta(hours=1)),"RUNNING","1000","1000")
        rows=[]
        for i in range(60):
            start=now.replace(second=0,microsecond=0)-timedelta(minutes=60-i); price=D("99.94")+D(i)*D(".001")
            rows.append(dict(source=config.source,instrument="BTC/USD",start_at=stamp(start),
                             end_at=stamp(start+timedelta(minutes=1)),available_at=stamp(now),
                             open=str(price),high=str(price+D(".002")),low=str(price-D(".001")),
                             close=str(price+D(".001")),volume="2"))
        state.candle_context=dict(rows=rows)
        snapshot=build_snapshot(config,state,Quote("q",stamp(now),D("100"),D("100.01"),config.source),now)
        generated=schema(snapshot["snapshot_id"],snapshot)
        self.assertEqual(set(generated),{"oneOf"})
        options=[{k:v["const"] for k,v in b["properties"].items()} for b in generated["oneOf"]]
        self.assertEqual(options,[dict(snapshot_id=snapshot["snapshot_id"],**o) for o in snapshot["decision_contract"]])
        for option in options:
            self.assertEqual(validate_answer(option,snapshot)["action"],option["action"])
        entry=next(o for o in options if o["action"]=="REQUEST_ENTRY")
        altered=dict(entry,action="WAIT")
        self.assertNotIn(altered,options)
        with self.assertRaises(ModelError) as caught:validate_answer(altered,snapshot)
        self.assertEqual(caught.exception.code,"CONTRACT_INVALID")
        self.assertEqual(validate_answer(dict(entry,evidence_refs=list(reversed(entry["evidence_refs"]))),snapshot)["action"],"REQUEST_ENTRY")

    def test_metadata_time_counts_toward_budget_and_late_result_is_rejected(self):
        now=utc_now()
        config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
        state=State("budget",stamp(now),stamp(now+timedelta(hours=1)),"RUNNING","1000","1000")
        snapshot=build_snapshot(config,state,Quote("q",stamp(now),D("100"),D("100.01"),config.source),now)
        wire=dict(snapshot_id=snapshot["snapshot_id"],**snapshot["decision_contract"][0])
        raw=dict(model=config.llm_model,done=True,done_reason="stop",message=dict(content=json.dumps(wire)))
        client=OllamaClient()
        with patch.object(client,"status",return_value=dict(digest=config.llm_digest)), \
                patch.object(client,"_request",return_value=raw) as request, \
                patch("traid_ai.qwen.time.monotonic",side_effect=[10,13,31]):
            with self.assertRaises(ModelError) as caught:client.decide(snapshot,digest=config.llm_digest)
        self.assertEqual(request.call_args.kwargs["timeout"],17)
        self.assertEqual(caught.exception.code,"MODEL_TIMEOUT")
