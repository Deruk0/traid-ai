from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from traid_ai.dashboard import SessionManager
from traid_ai.diagnostics import decision_summary
from traid_ai.engine import Engine
from traid_ai.health import technical_health
from traid_ai.models import Config, Quote, D, stamp, utc_now
from traid_ai.store import Store


def event(kind, payload):
    return dict(kind=kind, occurred_at=stamp(utc_now()), payload=payload)


def request(identity, *, eligible=False):
    return event("AI_REQUEST", dict(snapshot=dict(snapshot_id=identity,
        decision_contract=[dict(action="REQUEST_ENTRY" if eligible else "WAIT")],
        evidence=dict(c15_range_pct="0.20", round_trip_cost_pct="0.30"),
        candidates=[dict(action="REQUEST_ENTRY",status="ELIGIBLE" if eligible else "BLOCKED",
            conditions=dict(cost_scale=dict(passed=eligible,
                evidence_refs=["c15_range_pct","round_trip_cost_pct"])))])))


class DiagnosticsTests(unittest.TestCase):
    def test_code_block_is_distinct_from_model_declining_allowed_entry(self):
        events=[request("blocked"),request("eligible",eligible=True),
            event("AI_DECISION",dict(snapshot_id="blocked",accepted=True,action="WAIT",reason_code="COST_SCALE_TOO_SMALL")),
            event("AI_DECISION",dict(snapshot_id="eligible",accepted=True,action="WAIT",reason_code="CANDIDATE_DECLINED"))]
        result=decision_summary(events)
        self.assertEqual((result["blocked"],result["eligible"],result["requests_with_entry_option"]),(1,1,1))
        self.assertEqual(result["accepted_waits"],2)
        self.assertEqual(result["unanswered"],0)
        self.assertEqual(result["blocking_conditions"][0]["count"],1)
        self.assertEqual(result["blocking_conditions"][0]["latest_facts"]["round_trip_cost_pct"],"0.30")

    def test_rejected_unanswered_and_risk_refusal_are_separate(self):
        result=decision_summary([request("rejected"),request("unfinished"),
            event("AI_DECISION",dict(snapshot_id="rejected",accepted=False,action="WAIT")),
            event("RISK_VERDICT",dict(approved=False)),event("PAPER_FILL",dict(side="BUY"))])
        self.assertEqual((result["rejected"],result["unanswered"],result["risk_denials"],result["fills"]),(1,1,1,1))
        self.assertEqual(result["accepted_waits"],0)

    def test_legacy_requests_have_no_invented_blocking_conditions(self):
        result=decision_summary([event("AI_REQUEST",dict(snapshot=dict(snapshot_id="legacy",
                                         allowed_actions=["WAIT","REQUEST_ENTRY"])))])
        self.assertEqual(result["requests_with_entry_option"],1)
        self.assertEqual(result["blocked"],0)
        self.assertEqual(result["blocking_conditions"],[])

    def test_model_entry_is_distinct_from_broker_refusal_and_stale_answer(self):
        result=decision_summary([request("entry",eligible=True),request("late",eligible=True),
            event("AI_DECISION",dict(snapshot_id="entry",accepted=True,action="REQUEST_ENTRY",
                                     reason_code="ENTRY_REVIEW_ACCEPTED")),
            event("RISK_VERDICT",dict(approved=True,reason="WITHIN_LIMITS")),
            event("RISK_VERDICT",dict(approved=False,reason="BROKER_NOT_READY")),
            event("AI_DECISION",dict(snapshot_id="late",accepted=False,action="WAIT",
                                     reason="Снимок устарел или портфель изменился"))])
        self.assertEqual(result["accepted_entries"],1)
        self.assertEqual(result["fills"],0)
        self.assertEqual(result["risk_reason_counts"],{"BROKER_NOT_READY":1})
        self.assertEqual(result["rejected_reason_counts"],{"Снимок устарел или портфель изменился":1})

    def test_filter_precedes_truncation_and_preserves_complete_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"session.sqlite3"; store=Store(path)
            now=utc_now()
            engine=Engine.start(store,Config(source="synthetic"),now,now+timedelta(hours=1))
            events=[("ORDER_UPDATE",dict(status="filled",side="buy",filled_qty="1")),
                    ("BROKER_FILL",dict(side="BUY",quantity="1",price="100",fee="0",reason="TEST"))]
            events += [("AI_REQUEST",dict(snapshot=dict(snapshot_id=str(i),source="synthetic",
                observed_at=stamp(now),created_at=stamp(now),market=dict(mid="100",spread_bps="1")))) for i in range(600)]
            try:
                store.commit_step(engine.state,"noise",stamp(now),events)
            finally:
                store.close()
            manager=SessionManager(path)
            view=manager.view(engine.state.session_id)
            self.assertEqual(len(view["events"]),500)
            kinds={e["kind"] for e in view["important_events"]}
            self.assertEqual(kinds,{"SESSION_STARTED","ORDER_UPDATE","BROKER_FILL"})
            self.assertFalse(any(e["kind"] in {"AI_REQUEST","AI_DECISION","SNAPSHOT"} for e in view["important_events"]))
            self.assertEqual(len(manager.view(engine.state.session_id,full_events=True)["events"]),603)
            manager.shutdown()

    def test_inactive_pending_ai_is_not_reported_as_thinking_or_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/"session.sqlite3"); now=utc_now()
            config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
            engine=Engine.start(store,config,now,now+timedelta(hours=1))
            engine.state.ai=dict(status="PENDING",processed_at=stamp(now))
            result=technical_health(config,engine.state,[],now+timedelta(hours=2),active=False)
            self.assertTrue(result["recovery_required"])
            components={r["component"]:r for r in result["components"]}
            self.assertEqual(components["worker"]["status"],"RECOVERY_REQUIRED")
            self.assertEqual(components["qwen"]["status"],"INTERRUPTED")
            self.assertIsNone(components["qwen"]["waiting_seconds"])
            self.assertEqual(engine.state.status,"RUNNING")
            self.assertEqual(engine.state.ai["status"],"PENDING")
            engine.strategy.close(); store.close()
