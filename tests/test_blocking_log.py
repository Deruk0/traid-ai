"""Recorded blockers retain their owner, direction and evidence in the panel."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from traid_ai.diagnostics import blocking_log
from traid_ai.dashboard import SessionManager
from traid_ai.engine import Engine
from traid_ai.models import Config, stamp, utc_now
from traid_ai.store import Store


def event(seq, kind, **payload):
    return dict(seq=seq, kind=kind, occurred_at=f"2026-10-19T00:00:{seq:02d}+00:00", payload=payload)


class BlockingLogTests(unittest.TestCase):
    def test_required_filter_is_not_attributed_to_model_and_warning_is_not_blocker(self):
        snapshot = dict(snapshot_id="blocked", decision_contract=[dict(action="WAIT")],
            evidence=dict(trend="-1", cost="0.6"), candidates=[dict(action="REQUEST_ENTRY",
            conditions=dict(positive_trend=dict(passed=False, evidence_refs=["trend"]),
                            cost_scale=dict(passed=False, required=False, evidence_refs=["cost"])))])
        events = [event(1, "AI_REQUEST", snapshot=snapshot),
                  event(2, "AI_DECISION", snapshot_id="blocked", accepted=True, action="WAIT")]
        before = deepcopy(events)
        rows = blocking_log(events)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload"]["component"], "Фильтры кандидата")
        self.assertEqual(rows[0]["payload"]["facts"], {"trend": "-1"})
        self.assertEqual(events, before)

    def test_model_wait_and_validation_rejection_are_separate(self):
        rows = blocking_log([event(1, "AI_REQUEST", snapshot=dict(snapshot_id="entry",
            decision_contract=[dict(action="WAIT"), dict(action="REQUEST_ENTRY")])),
            event(2, "AI_DECISION", snapshot_id="entry", accepted=True, action="WAIT", reason="Ждём свечу"),
            event(3, "AI_DECISION", snapshot_id="late", accepted=False, action="WAIT",
                  error_code="SNAPSHOT_EXPIRED", reason="Снимок устарел")])
        self.assertEqual([r["payload"]["component"] for r in rows], ["Qwen", "Проверка ответа Qwen"])
        self.assertEqual(rows[0]["payload"]["scope"], "Вход")

    def test_exit_refusal_links_to_intent_and_unknown_direction_is_not_invented(self):
        rows = blocking_log([event(1, "DECISION", decision_id="exit", action="REQUEST_EXIT"),
            event(2, "RISK_VERDICT", decision_id="exit", approved=False, reason="BROKER_NOT_READY"),
            event(3, "RISK_VERDICT", approved=False, reason="BUDGET_OR_MINIMUM"),
            event(4, "RISK_VERDICT", action="REQUEST_EXIT", approved=False, reason="BROKER_NOT_READY")])
        self.assertEqual([r["payload"]["scope"] for r in rows], ["Выход", "Вход / выход", "Выход"])

    def test_holding_and_missing_quote_are_not_declared_execution_failures(self):
        rows = blocking_log([event(1, "DECISION", action="WAIT", reason="TRAILING_POSITION_HOLD"),
            event(2, "PROTECTIVE_EXIT_REQUIRED", reason="MAX_HOLD")])
        self.assertEqual(rows[0]["payload"]["status"], "Штатное удержание")
        self.assertEqual(rows[1]["payload"]["status"], "Требуется выход")

    def test_repeats_merge_but_trade_breaks_group_and_filter_precedes_limit(self):
        events = [event(1, "BROKER_UNRESOLVED", reason="Сверка"),
                  event(2, "BROKER_UNRESOLVED", reason="Сверка"),
                  event(3, "BROKER_FILL", side="BUY"),
                  event(4, "BROKER_UNRESOLVED", reason="Сверка")]
        events += [event(5, "SNAPSHOT", bid="1", ask="2")]*600
        rows = blocking_log(events)
        self.assertEqual([r["payload"]["count"] for r in rows], [2, 1])
        self.assertEqual(rows[0]["payload"]["source_seq"], 2)
        self.assertEqual(len(blocking_log(events, limit=1)), 1)

    def test_panel_reads_full_history_and_export_retains_all_blockers_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"panel.sqlite3"
            store = Store(path)
            now = utc_now()
            engine = Engine.start(store, Config(source="synthetic"), now, now+timedelta(hours=1))
            pairs = [("ORDER_ABORTED", dict(reason="AI_SNAPSHOT_EXPIRED_BEFORE_SUBMIT"))]
            pairs += [("SNAPSHOT", dict(bid="1", ask="2"))]*600
            store.commit_step(engine.state, "recorded", stamp(now), pairs)
            before = engine.state.to_dict()
            count = len(store.events(engine.state.session_id))
            store.close()
            manager = SessionManager(path)
            try:
                view = manager.view(engine.state.session_id)
                self.assertEqual(view["blocking_events"][0]["payload"]["scope"], "Вход")
                self.assertEqual(view["state"], before)
                self.assertEqual(view["event_count"], count)
                self.assertIn("trade-blocking-log-v1", view["capabilities"])
                self.assertEqual(len(manager.view(engine.state.session_id, full_events=True)["events"]), count)
            finally:
                manager.shutdown()
