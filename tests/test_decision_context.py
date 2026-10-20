from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from traid_ai.decision_context import (ENTRY_ID, EXIT_ID, build_decision_context,
                                       render_reason, validate_selection)
from traid_ai.models import Config, D, Quote, State, stamp


class DecisionContextTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
        self.config = Config(source="kraken-public",decision_policy="evidence-v1",exit_policy="fixed-stop-v1")
        self.state = State("test", stamp(self.now-timedelta(minutes=20)),
                           stamp(self.now+timedelta(seconds=660)), "RUNNING", "1000", "1000")
        self.quote = Quote("quote", stamp(self.now), D("100"), D("101"), "kraken-public")
        self.analytics = dict(costs=dict(estimated_round_trip_bps="60"),
                              quality=dict(volume_available=False, candle_history_available=False))
        self.candles = dict(status="READY", count=60, source="kraken-public", instrument="BTC/USD",
                            volume_available=False, windows=[dict(horizon_minutes=5, status="READY", return_pct="0.3", volume="2"),
                                                            dict(horizon_minutes=15, status="READY", return_pct="0.5", range_pct="1.2")])

    def context(self):
        return build_decision_context(self.config, self.state, self.quote, self.now, self.analytics, self.candles)

    def snapshot(self):
        return dict(snapshot_id="snapshot", allowed_actions=["WAIT", "REQUEST_ENTRY", "REQUEST_EXIT"], **self.context())

    def selection(self, snapshot, code):
        option = next(option for option in snapshot["decision_contract"] if option["reason_code"] == code)
        return dict(snapshot_id=snapshot["snapshot_id"], **deepcopy(option))

    def codes(self, context):
        return {option["reason_code"] for option in context["decision_contract"]}

    def test_eligible_candidate_can_be_selected_or_declined_without_risk_override(self):
        before = deepcopy((self.config.to_dict(), self.state.to_dict(), self.analytics, self.candles))
        snapshot = self.snapshot()
        candidate = snapshot["candidates"][0]
        self.assertEqual(candidate["candidate_id"], ENTRY_ID)
        self.assertEqual(candidate["status"], "ELIGIBLE")
        self.assertTrue(candidate["research_only"])
        self.assertIsNone(candidate["expected_return"])
        self.assertEqual(candidate["holding_horizon_seconds"], 300)
        self.assertTrue(all(c["passed"] for c in candidate["conditions"].values()))
        self.assertEqual(set(candidate["invalidation"]), set(candidate["conditions"]))
        self.assertEqual(self.codes(snapshot), {"ENTRY_CRITERIA_MET", "CANDIDATE_DECLINED"})
        for code in ("ENTRY_CRITERIA_MET", "CANDIDATE_DECLINED"):
            result = validate_selection(self.selection(snapshot, code), snapshot)
            self.assertEqual(result["explanation_status"], "VERIFIED")
            self.assertEqual(result["reason_code"], code)
            self.assertNotIn("reason", self.selection(snapshot, code))
        self.assertEqual(before, (self.config.to_dict(), self.state.to_dict(), self.analytics, self.candles))

    def test_positive_volume_cannot_be_explained_as_missing(self):
        snapshot = self.snapshot()
        quality = snapshot["data_quality"]
        self.assertTrue(quality["volume_available"])
        self.assertTrue(quality["candles_available"])
        self.assertEqual(quality["volume_state"], "AVAILABLE")
        self.assertNotIn("VOLUME_UNAVAILABLE", self.codes(snapshot))
        answer = self.selection(snapshot, "CANDIDATE_DECLINED")
        answer.update(reason_code="VOLUME_UNAVAILABLE", evidence_refs=["c5_status", "c5_volume_btc"])
        with self.assertRaises(ValueError):
            validate_selection(answer, snapshot)

    def test_ready_one_minute_volume_is_present_even_while_entry_window_warms(self):
        self.candles.update(status="WARMUP",count=1,windows=[
            dict(horizon_minutes=1,status="READY",volume="2"),
            dict(horizon_minutes=5,status="WARMUP"),dict(horizon_minutes=15,status="WARMUP")])
        snapshot=self.snapshot()
        self.assertTrue(snapshot["data_quality"]["volume_available"])
        self.assertEqual(snapshot["data_quality"]["volume_state"],"INCOMPLETE")
        self.assertEqual(snapshot["data_quality"]["volume_window_minutes"],5)
        self.assertEqual(self.codes(snapshot),{"CANDLES_INCOMPLETE"})

    def test_zero_volume_is_measured_and_distinct_from_missing(self):
        self.candles["windows"][0]["volume"] = "0"
        snapshot = self.snapshot()
        quality = snapshot["data_quality"]
        self.assertTrue(quality["volume_available"])
        self.assertEqual(quality["volume_state"], "ZERO")
        self.assertIn("ZERO_VOLUME", self.codes(snapshot))
        self.assertNotIn("VOLUME_UNAVAILABLE", self.codes(snapshot))
        reason = validate_selection(self.selection(snapshot, "ZERO_VOLUME"), snapshot)["reason"]
        self.assertIn("0 BTC", reason)
        self.candles["windows"][0]["volume"] = None
        snapshot = self.snapshot()
        self.assertFalse(snapshot["data_quality"]["volume_available"])
        self.assertEqual(snapshot["data_quality"]["volume_state"], "UNAVAILABLE")
        self.assertIn("VOLUME_UNAVAILABLE", self.codes(snapshot))
        self.assertNotIn("ZERO_VOLUME", self.codes(snapshot))

    def test_unavailable_stale_and_incomplete_have_truthful_distinct_options(self):
        self.candles.update(status="UNAVAILABLE", count=0, windows=[])
        snapshot = self.snapshot()
        self.assertEqual(self.codes(snapshot), {"CANDLES_UNAVAILABLE"})
        self.assertFalse(snapshot["data_quality"]["candles_available"])
        self.candles.update(status="STALE", count=60,
                            windows=[dict(horizon_minutes=m, status="STALE") for m in (5, 15)])
        snapshot = self.snapshot()
        self.assertEqual(self.codes(snapshot), {"CANDLES_STALE"})
        self.assertEqual(snapshot["data_quality"]["volume_state"], "STALE")
        self.candles.update(status="WARMUP", count=2,
                            windows=[dict(horizon_minutes=m, status="WARMUP") for m in (5, 15)])
        snapshot = self.snapshot()
        self.assertEqual(self.codes(snapshot), {"CANDLES_INCOMPLETE"})
        self.assertEqual(snapshot["data_quality"]["volume_state"], "INCOMPLETE")

    def test_global_hour_warmup_does_not_block_ready_5_and_15_minute_candidate(self):
        self.candles.update(status="WARMUP", count=15)
        self.assertEqual(self.context()["candidates"][0]["status"], "ELIGIBLE")

    def test_multiple_true_blockers_are_all_selectable(self):
        self.candles["windows"][0].update(return_pct="-0.1", volume="0")
        self.candles["windows"][1]["range_pct"] = "0.2"
        self.state.deadline_at = stamp(self.now+timedelta(seconds=100))
        self.state.status = "PAUSED_ENTRIES"
        snapshot = self.snapshot()
        self.assertEqual(self.codes(snapshot), {"ZERO_VOLUME", "TREND_NOT_CONFIRMED", "COST_SCALE_TOO_SMALL",
                                                "ENTRY_WINDOW_TOO_SHORT", "ENTRIES_DISABLED"})
        self.assertEqual(snapshot["candidates"][0]["status"], "BLOCKED")
        for code in self.codes(snapshot):
            self.assertEqual(validate_selection(self.selection(snapshot, code), snapshot)["reason_code"], code)

    def test_rule_thresholds_are_strict_except_minimum_holding_time(self):
        self.state.deadline_at = stamp(self.now+timedelta(seconds=360))
        self.candles["windows"][1]["range_pct"] = "0.600001"
        self.assertEqual(self.context()["candidates"][0]["status"], "ELIGIBLE")
        self.candles["windows"][1]["range_pct"] = "0.6"
        self.assertIn("COST_SCALE_TOO_SMALL", self.codes(self.context()))
        self.candles["windows"][0]["return_pct"] = "0"
        self.assertIn("TREND_NOT_CONFIRMED", self.codes(self.context()))
        self.assertNotIn("ENTRY_WINDOW_TOO_SHORT", self.codes(self.context()))
        self.state.deadline_at = stamp(self.now+timedelta(seconds=359))
        self.assertIn("ENTRY_WINDOW_TOO_SHORT", self.codes(self.context()))

    def test_volume_is_incomplete_only_when_required_five_minute_window_is_incomplete(self):
        self.candles["windows"][0] = dict(horizon_minutes=5, status="GAPS", volume=None)
        self.candles["status"] = "GAPS"
        snapshot = self.snapshot()
        self.assertEqual(snapshot["data_quality"]["volume_state"], "INCOMPLETE")
        self.assertEqual(snapshot["data_quality"]["usable_horizons_minutes"], [15])
        self.assertIn("CANDLES_INCOMPLETE", self.codes(snapshot))
        self.assertNotIn("VOLUME_UNAVAILABLE", self.codes(snapshot))

    def test_position_can_be_monitored_or_reviewed_for_exit_without_market_history(self):
        self.state.quantity = "0.123"
        self.state.status = "FLATTENING"
        self.candles.update(status="UNAVAILABLE", count=0, windows=[])
        snapshot = self.snapshot()
        self.assertEqual(snapshot["candidates"][0]["status"], "NOT_APPLICABLE")
        self.assertEqual(snapshot["candidates"][1]["candidate_id"], EXIT_ID)
        self.assertEqual(self.codes(snapshot), {"POSITION_MONITORING", "POSITION_REVIEW"})
        for code in self.codes(snapshot):
            result = validate_selection(self.selection(snapshot, code), snapshot)
            self.assertIn("0.123 BTC", result["reason"])
            self.assertNotIn("нет данных", result["reason"])

    def test_pending_order_blocks_entry_and_is_explained_by_authority_facts(self):
        self.state.broker = {"pending": {"phase": "SUBMITTING"}}
        snapshot = self.snapshot()
        self.assertEqual(self.codes(snapshot), {"ENTRIES_DISABLED"})
        self.assertTrue(snapshot["evidence"]["pending_order"])
        self.assertIn("да", validate_selection(self.selection(snapshot, "ENTRIES_DISABLED"), snapshot)["reason"])

    def test_exact_wire_keys_exclude_free_form_or_extra_reason(self):
        snapshot = self.snapshot()
        valid = self.selection(snapshot, "CANDIDATE_DECLINED")
        for field in valid:
            invalid = deepcopy(valid)
            del invalid[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_selection(invalid, snapshot)
        invalid = dict(valid, reason="Объём недоступен")
        with self.assertRaises(ValueError):
            validate_selection(invalid, snapshot)

    def test_duplicates_unknown_missing_and_wrong_type_references_rejected(self):
        snapshot = self.snapshot()
        valid = self.selection(snapshot, "CANDIDATE_DECLINED")
        refs = valid["evidence_refs"]
        for invalid_refs in (refs+[refs[0]], refs+["invented"], refs[:-1], [], "candle_count", [None], [{}]):
            invalid = dict(valid, evidence_refs=invalid_refs)
            with self.subTest(refs=invalid_refs), self.assertRaises(ValueError):
                validate_selection(invalid, snapshot)
        valid["evidence_refs"] = list(reversed(refs))
        self.assertEqual(validate_selection(valid, snapshot)["evidence_refs"], refs)

    def test_snapshot_candidate_reason_and_allowed_action_must_match(self):
        snapshot = self.snapshot()
        valid = self.selection(snapshot, "ENTRY_CRITERIA_MET")
        for patch in (dict(snapshot_id="other"), dict(snapshot_id=None), dict(action="WAIT"),
                      dict(candidate_id=EXIT_ID), dict(reason_code="NO_PROVEN_EDGE_OR_INSUFFICIENT_EVIDENCE")):
            invalid = dict(valid, **patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_selection(invalid, snapshot)
        snapshot["allowed_actions"] = ["WAIT"]
        with self.assertRaises(ValueError):
            validate_selection(valid, snapshot)

    def test_reason_is_bounded_factual_and_does_not_predict_profit(self):
        self.candles["windows"][1]["range_pct"] = "0.200000"
        snapshot = self.snapshot()
        result = validate_selection(self.selection(snapshot, "COST_SCALE_TOO_SMALL"), snapshot)
        self.assertIn("0.2 %", result["reason"])
        self.assertIn("0.6 %", result["reason"])
        self.assertIn("не превышает", result["reason"])
        self.assertNotIn("прибы", result["reason"].lower())
        self.assertLessEqual(len(result["reason"]), 240)
        self.assertEqual(snapshot["evidence"]["round_trip_cost_pct"], "0.6")
        self.assertEqual(snapshot["evidence"]["seconds_to_flatten"], "600.0")
        for option in snapshot["decision_contract"]:
            self.assertLessEqual(len(render_reason(option["reason_code"], snapshot["evidence"])), 240)
        with self.assertRaises(ValueError):
            render_reason("UNKNOWN", snapshot["evidence"])

    def test_context_does_not_cross_sources(self):
        self.candles["source"] = "alpaca-paper"
        with self.assertRaises(ValueError):
            self.context()


if __name__ == "__main__":
    unittest.main()
