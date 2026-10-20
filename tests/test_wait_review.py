from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from traid_ai.decision_context import validate_selection
from traid_ai.diagnostics import decision_summary
from traid_ai.engine import Engine
from traid_ai.evaluation import compare_session
from traid_ai.models import Config, D, Quote, encode, parse_time, stamp, utc_now
from traid_ai.qwen import QwenStrategy, build_snapshot
from traid_ai.store import Store
from test_alpaca import await_condition
from test_evidence_qwen import EvidenceModel


class WaitReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/"state.sqlite3")
        self.now=utc_now().replace(second=0,microsecond=0)
        self.start=self.now
        self.config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64,
                           exit_policy="fixed-stop-v1")
        self.engine=Engine.start(self.store,self.config,self.now,self.now+timedelta(hours=1))
        self.client=EvidenceModel(action="WAIT")
        self.engine.strategy.close(); self.engine.strategy=QwenStrategy(client=self.client)
        self.engine.state.history=["100"]*5
        self.rows=[]
        for i in range(60):
            start=self.now-timedelta(minutes=60-i); price=D("99.94")+D(i)*D(".001")
            self.rows.append(dict(source="kraken-public",instrument="BTC/USD",start_at=stamp(start),
                end_at=stamp(start+timedelta(minutes=1)),available_at=stamp(self.now),
                open=str(price),high=str(price+D(".002")),low=str(price-D(".001")),
                close=str(price+D(".001")),volume="2"))
        self.engine.state.candle_context=dict(rows=self.rows)
        self.save("ready")

    def tearDown(self):
        self.engine.strategy.close(); self.store.close(); self.tmp.cleanup()

    def save(self, identity):
        self.store.commit_step(self.engine.state,identity,stamp(self.now),[])

    def tick(self, seconds=1, **kwargs):
        self.now+=timedelta(seconds=seconds)
        return self.engine.step(self.now,Quote(stamp(self.now),stamp(self.now),D("100"),D("100.01"),
                                              "kraken-public"),**kwargs)

    def done(self):
        await_condition(lambda:self.engine.strategy.job and self.engine.strategy.job["done"])

    def snapshot(self, config=None):
        return build_snapshot(config or self.config,self.engine.state,
            Quote("q",stamp(self.now),D("100"),D("100.01"),"kraken-public"),self.now)

    def test_cost_warning_is_visible_but_does_not_remove_entry(self):
        s=self.snapshot()
        self.assertEqual(s["snapshot_version"],"market-summary-v6")
        candidate=s["candidates"][0]
        self.assertEqual(candidate["status"],"ELIGIBLE")
        self.assertFalse(candidate["conditions"]["cost_scale"]["passed"])
        self.assertFalse(candidate["conditions"]["cost_scale"]["required"])
        self.assertNotIn("cost_scale",candidate["invalidation"])
        self.assertIn("REQUEST_ENTRY",s["allowed_actions"])
        self.assertIn("COST_SCALE_TOO_SMALL",{o["reason_code"] for o in s["decision_contract"]})
        self.assertNotIn("CANDIDATE_DECLINED",{o["reason_code"] for o in s["decision_contract"]})
        self.assertEqual(self.config.fee_rate,D(".001"))

    def test_old_explicit_policy_is_unchanged_and_does_not_get_review_fields(self):
        old=Config.from_dict(replace(self.config,decision_policy="evidence-v1").to_dict())
        s=self.snapshot(old)
        self.assertEqual(s["snapshot_version"],"market-summary-v5")
        self.assertEqual(s["candidates"][0]["status"],"BLOCKED")
        self.assertNotIn("REQUEST_ENTRY",s["allowed_actions"])
        self.assertNotIn("review_policy",s)

    def test_entry_still_requires_measured_volume_trend_time_and_authority(self):
        for field,value in (("volume","0"),("close","99")):
            rows=deepcopy(self.rows)
            if field=="volume":
                for row in rows[-5:]:row[field]=value
            else:
                for row in rows[-5:]:row.update(open="99",close="99",high="99.1",low="98.9")
            self.engine.state.candle_context=dict(rows=rows)
            self.assertNotIn("REQUEST_ENTRY",self.snapshot()["allowed_actions"])
        self.engine.state.candle_context=dict(rows=self.rows)
        self.engine.state.status="PAUSED_ENTRIES"
        self.assertNotIn("REQUEST_ENTRY",self.snapshot()["allowed_actions"])
        self.engine.state.status="RUNNING"
        self.engine.state.deadline_at=stamp(self.now+timedelta(minutes=2))
        self.assertNotIn("REQUEST_ENTRY",self.snapshot()["allowed_actions"])

    def test_missing_and_stale_candles_remain_blocked(self):
        self.engine.state.candle_context={}
        self.assertNotIn("REQUEST_ENTRY",self.snapshot()["allowed_actions"])
        self.engine.state.candle_context=dict(rows=self.rows)
        self.now+=timedelta(minutes=4)
        self.assertNotIn("REQUEST_ENTRY",self.snapshot()["allowed_actions"])

    def test_entry_below_old_cost_filter_still_passes_risk_and_charges_fees(self):
        self.client.action="REQUEST_ENTRY"
        self.tick();self.done();events=self.tick()
        self.assertEqual(self.engine.state.entries,1)
        self.assertEqual(self.engine.state.ai["reason_code"],"ENTRY_REVIEW_ACCEPTED")
        self.assertTrue(any(k=="RISK_VERDICT" and p["approved"] for k,p in events))
        self.assertGreater(D(self.engine.state.fees),0)
        self.assertLessEqual(D(self.engine.state.entry_cost),self.config.capital*self.config.max_exposure)
        self.assertEqual(self.engine.state.ai["prompt_version"],"qwen-trader-v6")

    def test_wait_has_deadline_and_timer_reassesses_without_automatic_entry(self):
        self.tick();self.done();self.tick()
        review_at=self.engine.state.ai["review_at"]
        self.assertEqual(review_at,stamp(self.start+timedelta(seconds=61)))
        self.assertEqual(self.engine.state.ai["review_trigger"],"NEXT_CLOSED_CANDLE_OR_TIMER")
        events=self.tick(10)
        self.assertEqual(len(self.client.calls),1)
        self.assertTrue(any(k=="DECISION" and p["reason"]=="AI_REVIEW_SCHEDULED" for k,p in events))
        self.assertEqual(self.engine.state.ai["review_at"],review_at)
        self.tick(20);self.tick(20);events=self.tick(10);self.done();self.tick()
        self.assertTrue(any(k=="AI_REVIEW_DUE" and p["trigger"]=="TIMER" for k,p in events))
        self.assertEqual(len(self.client.calls),2)
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(self.engine.state.ai["wait_streak"],2)

    def test_new_closed_candle_can_trigger_review_before_timer(self):
        self.now+=timedelta(seconds=40)
        self.tick();self.done();self.tick()
        review_at=self.engine.state.ai["review_at"]
        self.now=self.start+timedelta(seconds=64)
        row=dict(self.rows[-1],start_at=stamp(self.start),end_at=stamp(self.start+timedelta(minutes=1)),
                 available_at=stamp(self.now))
        self.engine.state.candle_context=dict(rows=self.rows+[row]);self.save("new-candle")
        events=self.tick();self.done()
        self.assertLess(self.now,parse_time(review_at))
        self.assertTrue(any(k=="AI_REVIEW_DUE" and p["trigger"]=="CANDLE_CLOSED" for k,p in events))
        self.assertEqual(len(self.client.calls),2)
        self.assertEqual(self.engine.state.entries,0)

    def test_restart_preserves_wait_deadline(self):
        self.tick();self.done();self.tick()
        expected=self.engine.state.ai["review_at"]
        self.engine.strategy.close()
        client=EvidenceModel(action="WAIT")
        self.engine=Engine(self.store,self.engine.state.session_id)
        self.engine.strategy.close();self.engine.strategy=QwenStrategy(client=client)
        self.tick(10)
        self.assertEqual(client.calls,[])
        self.assertEqual(self.engine.state.ai["review_at"],expected)
        self.tick(20);self.tick(20);self.tick(10);self.done()
        self.assertEqual(len(client.calls),1)

    def test_model_cannot_set_its_own_wait_deadline_or_use_generic_refusal(self):
        s=self.snapshot();choice=next(c for c in s["decision_contract"] if c["action"]=="WAIT")
        wire=dict(snapshot_id=s["snapshot_id"],**choice)
        self.assertIn("review_at",validate_selection(wire,s))
        for patch in (dict(review_at="2099-01-01T00:00:00+00:00"),dict(reason_code="CANDIDATE_DECLINED")):
            with self.assertRaises(ValueError):validate_selection(dict(wire,**patch),s)
        s["review_policy"]["review_at"]=stamp(self.now+timedelta(hours=1))
        with self.assertRaises(ValueError):validate_selection(wire,s)

    def test_stop_and_deadline_override_wait_schedule(self):
        self.tick();self.done();self.tick()
        self.tick(1,stop=True)
        self.assertEqual(self.engine.state.status,"CLOSED")
        self.assertEqual(self.engine.state.entries,0)
        self.assertEqual(len(self.client.calls),1)

    def test_deadline_exits_wait_without_new_model_request(self):
        self.tick();self.done();self.tick()
        self.now=parse_time(self.engine.state.deadline_at)-timedelta(seconds=self.config.flatten_buffer_seconds)
        self.tick()
        self.assertEqual(self.engine.state.status,"CLOSED")
        self.assertEqual(self.engine.state.exit_reason,"DEADLINE")
        self.assertEqual(len(self.client.calls),1)

    def test_model_entry_still_rejected_by_independent_risk_limit(self):
        config=replace(self.config,max_exposure=D(".000001"))
        with self.store.db:
            self.store.db.execute("UPDATE sessions SET config=? WHERE id=?",
                                  (encode(config.to_dict()),self.engine.state.session_id))
        self.client.action="REQUEST_ENTRY"
        self.tick();self.done();events=self.tick()
        self.assertEqual(self.engine.state.ai["action"],"REQUEST_ENTRY")
        self.assertEqual(self.engine.state.entries,0)
        self.assertTrue(any(k=="RISK_VERDICT" and not p["approved"] for k,p in events))

    def test_diagnostics_separates_warning_from_entry_block(self):
        self.tick();self.done();self.tick()
        summary=decision_summary(self.store.events(self.engine.state.session_id))
        self.assertEqual(summary["requests_with_entry_option"],1)
        self.assertEqual(summary["blocking_conditions"],[])
        self.assertEqual(summary["warnings"][0]["code"],"cost_scale")

    def test_v6_replay_matches_policy_and_rejects_older_contract(self):
        self.client.action="REQUEST_ENTRY"
        self.tick();self.done();self.tick()
        export=dict(config=self.config.to_dict(),state=self.engine.state.to_dict(),
                    events=self.store.events(self.engine.state.session_id))
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],1)
        export["config"]["decision_policy"]="evidence-v1"
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],0)
