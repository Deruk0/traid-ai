from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from traid_ai.engine import Engine
from traid_ai.models import Config,D,Quote,stamp,utc_now
from traid_ai.qwen import MODEL,QwenStrategy,OllamaClient,build_snapshot,schema
from traid_ai.store import Store
from test_alpaca import await_condition


class EvidenceModel:
    def __init__(self,action="REQUEST_ENTRY",invalid=False):
        self.action,self.invalid=action,invalid
        self.calls=[]
    def decide(self,snapshot,*,digest,timeout=20):
        self.calls.append(snapshot)
        choice=next((c for c in snapshot["decision_contract"] if c["action"]==self.action),
                    next(c for c in snapshot["decision_contract"] if c["action"]=="WAIT"))
        answer=dict(snapshot_id=snapshot["snapshot_id"],**choice,latency_ms=10)
        if self.invalid:
            answer.update(reason_code="VOLUME_UNAVAILABLE",evidence_refs=["c5_status","c5_volume_btc"])
        return answer


class EvidenceQwenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/"state.sqlite3")
        self.now=utc_now().replace(second=0,microsecond=0)
        self.config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64,
                           decision_policy="evidence-v1")
        self.engine=Engine.start(self.store,self.config,self.now,self.now+timedelta(hours=1))
        self.engine.state.history=["100"]*5
        self.rows=[]
        for i in range(60):
            start=self.now-timedelta(minutes=60-i)
            price=D("97")+D(i)*D(".05")
            self.rows.append(dict(source="kraken-public",instrument="BTC/USD",start_at=stamp(start),
                end_at=stamp(start+timedelta(minutes=1)),available_at=stamp(self.now),
                open=str(price),high=str(price+D(".06")),low=str(price-D(".01")),
                close=str(price+D(".05")),volume="2"))
        self.engine.state.candle_context=dict(rows=self.rows)
        self.store.commit_step(self.engine.state,"warm",stamp(self.now),[])
        self.client=EvidenceModel()
        self.engine.strategy.close()
        self.engine.strategy=QwenStrategy(client=self.client)
    def tearDown(self):
        self.engine.strategy.close();self.store.close();self.tmp.cleanup()
    def tick(self):
        self.now+=timedelta(seconds=1)
        return self.engine.step(self.now,Quote(stamp(self.now),stamp(self.now),D("100"),D("100.01"),"kraken-public"))
    def done(self):
        await_condition(lambda:self.engine.strategy.job and self.engine.strategy.job["done"])

    def test_eligible_entry_keeps_risk_gate_and_persists_verified_facts(self):
        self.tick();self.done();events=self.tick()
        self.assertEqual(self.engine.state.entries,1)
        self.assertTrue(any(k=="RISK_VERDICT" and p["approved"] for k,p in events))
        self.assertLessEqual(D(self.engine.state.entry_cost),D("250"))
        ai=self.engine.state.ai
        self.assertEqual(ai["explanation_status"],"VERIFIED")
        self.assertEqual(ai["reason_code"],"ENTRY_CRITERIA_MET")
        self.assertEqual(ai["evidence"]["c5_volume_btc"],"10.000000")
        saved=self.store.load(self.engine.state.session_id)[1]
        self.assertEqual(saved.ai["evidence"],ai["evidence"])
        self.assertEqual(saved.ai["prompt_version"],"qwen-trader-v5")

    def test_false_missing_volume_reason_cannot_authorize_entry(self):
        self.client.invalid=True
        self.tick();self.done();self.tick()
        self.assertTrue(self.client.calls[0]["data_quality"]["volume_available"])
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["status"],"REJECTED")
        self.assertNotEqual(self.engine.state.ai.get("explanation_status"),"VERIFIED")

    def test_blocked_candidate_explains_costs_with_actual_values(self):
        self.engine.config=replace(self.config,fee_rate=D(".02"))
        with self.store.db:
            from traid_ai.models import encode
            self.store.db.execute("UPDATE sessions SET config=? WHERE id=?",
                (encode(self.engine.config.to_dict()),self.engine.state.session_id))
        self.tick();self.done();self.tick()
        snapshot=self.client.calls[0]
        self.assertNotIn("REQUEST_ENTRY",snapshot["allowed_actions"])
        self.assertEqual(self.engine.state.ai["reason_code"],"COST_SCALE_TOO_SMALL")
        self.assertIn(snapshot["evidence"]["c15_range_pct"].rstrip("0"),self.engine.state.ai["reason"])
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["explanation_status"],"VERIFIED")

    def test_context_has_one_authoritative_volume_definition(self):
        self.tick();self.done()
        snapshot=self.client.calls[0]
        self.assertEqual(snapshot["snapshot_version"],"market-summary-v5")
        self.assertTrue(snapshot["data_quality"]["volume_available"])
        self.assertNotIn("volume_available",snapshot["analytics"]["quality"])
        self.assertNotIn("candle_history_available",snapshot["analytics"]["quality"])
        self.assertNotIn("volume_available",snapshot["candles"])
        self.assertNotIn("reference_signal",snapshot["analytics"])
        self.assertNotIn("options",snapshot)
        self.assertNotIn("VOLUME_UNAVAILABLE",[c["reason_code"] for c in snapshot["decision_contract"]])

    def test_schema_and_ollama_transport_preserve_typed_evidence(self):
        quote=Quote("quote",stamp(self.now),D("100"),D("100.01"),"kraken-public")
        snapshot=build_snapshot(self.config,self.engine.state,quote,self.now)
        self.client.action="WAIT"
        answer=self.client.decide(snapshot,digest="a"*64)
        answer.pop("latency_ms")
        client=OllamaClient()
        with patch.object(client,"status",return_value=dict(digest="a"*64)),patch.object(client,"_request",
            return_value=dict(model=MODEL,done=True,message=dict(content=json.dumps(answer)))) as request:
            parsed=client.decide(snapshot,digest="a"*64)
        payload=request.call_args.args[1]
        for branch in payload["format"]["oneOf"]:
            self.assertNotIn("reason",branch["properties"])
            self.assertIn("reason_code",branch["required"])
        self.assertNotIn("tools",payload)
        self.assertEqual(parsed["explanation_status"],"VERIFIED")
        self.assertNotIn("latency_ms",answer)

    def test_new_features_reach_model_and_survive_store_restore(self):
        self.tick();self.done()
        snapshot=self.client.calls[0]
        windows={w["horizon_minutes"]:w for w in snapshot["candles"]["windows"]}
        self.assertEqual(snapshot["candles"]["version"],"candle-analytics-v2")
        self.assertEqual(windows[5]["movement"]["path_efficiency_pct"],"100.000000")
        self.assertEqual(windows[5]["movement"]["largest_step_share_pct"],"20.000000")
        self.assertEqual(snapshot["candles"]["relative_volume"]["ratio"],"1.000000")
        self.assertEqual(snapshot["candles"]["relative_volume"]["status"],"READY")
        saved=next(e["payload"]["snapshot"] for e in self.store.events(self.engine.state.session_id)
                   if e["kind"]=="AI_REQUEST")
        self.assertEqual(saved,snapshot)
        cfg,state=self.store.load(self.engine.state.session_id)
        quote=Quote("restore",stamp(self.now),D("100"),D("100.01"),"kraken-public")
        rebuilt=build_snapshot(cfg,state,quote,self.now)
        self.assertEqual(rebuilt["candles"],snapshot["candles"])

    def test_new_features_respect_available_at_and_do_not_change_entry_conditions(self):
        quote=Quote("quote",stamp(self.now),D("100"),D("100.01"),"kraken-public")
        before=build_snapshot(self.config,self.engine.state,quote,self.now)
        self.rows[-20]["available_at"]=stamp(self.now+timedelta(minutes=1))
        after=build_snapshot(self.config,self.engine.state,quote,self.now)
        self.assertEqual(after["candles"]["relative_volume"]["status"],"GAPS")
        self.assertIsNone(after["candles"]["relative_volume"]["ratio"])
        self.assertEqual(after["candles"]["count"],59)
        # 5/15-minute facts are still complete; the new baseline adds no trading gate.
        self.assertEqual(before["decision_contract"],after["decision_contract"])
        self.assertEqual(before["candidates"],after["candidates"])
        for row in self.rows:
            row["volume"]="0.00000001"
        tiny=build_snapshot(self.config,self.engine.state,quote,self.now)
        self.assertEqual(tiny["data_quality"]["volume_state"],"AVAILABLE")
        self.assertGreater(D(tiny["evidence"]["c5_volume_btc"]),0)

    def test_old_session_config_keeps_legacy_contract_on_restore(self):
        old=self.config.to_dict();old.pop("decision_policy")
        legacy=Config.from_dict(old)
        self.assertEqual(legacy.decision_policy,"legacy")
        quote=Quote("quote",stamp(self.now),D("100"),D("100.01"),"kraken-public")
        snapshot=build_snapshot(legacy,self.engine.state,quote,self.now)
        self.assertEqual(snapshot["snapshot_version"],"market-summary-v3")
        self.assertNotIn("decision_contract",snapshot)
        self.assertEqual(snapshot["candles"]["version"],"candle-analytics-v1")
        self.assertNotIn("relative_volume",snapshot["candles"])
        self.assertTrue(all("movement" not in w for w in snapshot["candles"]["windows"]))
        self.assertIn("reason",schema(snapshot["snapshot_id"],snapshot)["required"])
        for row in self.rows:
            row["volume"]="0.00000001"
        snapshot=build_snapshot(legacy,self.engine.state,quote,self.now)
        self.assertEqual(snapshot["candles"]["windows"][1]["volume"],"0.000000")

    def test_recorded_v5_evidence_is_validated_during_replay(self):
        from traid_ai.evaluation import compare_session
        self.tick();self.done();self.tick()
        export=dict(config=self.config.to_dict(),state=self.engine.state.to_dict(),
                    events=self.store.events(self.engine.state.session_id))
        report=compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"],1)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"],1)
        for event in export["events"]:
            if event["kind"]=="AI_DECISION":
                event["payload"]["answer"]["reason_code"]="VOLUME_UNAVAILABLE"
        report=compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"],0)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"],0)

    def test_archived_v4_still_replays_and_unknown_snapshot_version_is_rejected(self):
        from copy import deepcopy
        import hashlib
        from traid_ai.evaluation import compare_session
        from traid_ai.models import encode
        self.tick();self.done();self.tick()
        export=dict(config=self.config.to_dict(),state=self.engine.state.to_dict(),
                    events=self.store.events(self.engine.state.session_id))
        for version,expected in (("market-summary-v4",1),("market-summary-v999",0)):
            archive=deepcopy(export)
            request=next(e["payload"] for e in archive["events"] if e["kind"]=="AI_REQUEST")
            snapshot=request["snapshot"]
            snapshot["snapshot_version"]=version
            snapshot["candles"]["version"]="candle-analytics-v1"
            snapshot["candles"].pop("relative_volume")
            for window in snapshot["candles"]["windows"]:
                window.pop("movement")
            old_id=snapshot.pop("snapshot_id")
            snapshot["snapshot_id"]=hashlib.sha256(encode(snapshot).encode()).hexdigest()
            request["prompt_version"]="qwen-trader-v4"
            for event in archive["events"]:
                if event["kind"]=="AI_DECISION" and event["payload"]["snapshot_id"]==old_id:
                    event["payload"]["snapshot_id"]=snapshot["snapshot_id"]
                    event["payload"]["answer"]["snapshot_id"]=snapshot["snapshot_id"]
            self.assertEqual(compare_session(archive)["qwen_replay"]["replayed_decisions"],expected)


if __name__=="__main__":unittest.main()
