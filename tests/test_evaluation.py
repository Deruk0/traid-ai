from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from traid_ai.evaluation import compare_session
from traid_ai.models import Config, D, State, encode, stamp
from traid_ai.qwen import build_snapshot
from traid_ai.models import Quote


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DIGEST = "a" * 64


def dataset(prices=("100", "100", "100", "100", "100", "101", "99", "98"), *, qwen=False):
    config = Config(source="kraken-public", strategy_version="qwen-local-v1" if qwen else "demo-momentum-v1",
                    llm_digest=DIGEST if qwen else "",decision_policy="legacy")
    state = State("original", stamp(NOW), stamp(NOW + timedelta(hours=1)), "RUNNING", "1000", "1000")
    events = [dict(seq=1, kind="SESSION_STARTED", occurred_at=stamp(NOW), payload=config.to_dict())]
    for index, price in enumerate(prices):
        when = NOW + timedelta(seconds=index + 1)
        quote = Quote(str(index), stamp(when), D(price), D(price) + D(".01"), config.source)
        events.append(dict(seq=len(events) + 1, kind="SNAPSHOT", occurred_at=stamp(when), payload=asdict(quote)))
    return dict(config=config.to_dict(), state=state.to_dict(), events=events)


def add_decision(export, *, request_index=5, answer_index=6, action="REQUEST_ENTRY"):
    config = Config.from_dict(export["config"])
    state = State(**export["state"])
    quote_event = export["events"][request_index]
    raw = quote_event["payload"]
    quote = Quote(raw["quote_id"], raw["observed_at"], D(raw["bid"]), D(raw["ask"]), raw["source"])
    snapshot = build_snapshot(config, state, quote, datetime.fromisoformat(quote_event["occurred_at"]))
    request = dict(kind="AI_REQUEST", occurred_at=quote_event["occurred_at"], payload=dict(snapshot=snapshot, model_digest=DIGEST))
    answer_time = export["events"][answer_index]["occurred_at"]
    answer = dict(kind="AI_DECISION", occurred_at=answer_time,
                  payload=dict(snapshot_id=snapshot["snapshot_id"], accepted=True, action=action,
                               processed_at=answer_time, answer=dict(snapshot_id=snapshot["snapshot_id"], action=action,
                                                                      reason="Проверка")))
    export["events"] += [request, answer]
    export["events"].sort(key=lambda e: e["occurred_at"])
    return request, answer


class EvaluationTests(unittest.TestCase):
    def test_new_policy_cannot_replay_legacy_answer_without_evidence(self):
        export=dataset(qwen=True)
        add_decision(export)
        export["config"]["decision_policy"]="evidence-v1"
        report=compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"],0)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"],0)

    def test_reproducible_costs_and_no_network_or_mutation(self):
        export = dataset()
        original = deepcopy(export)
        with patch("traid_ai.qwen.OllamaClient", side_effect=AssertionError("No Ollama")):
            first = compare_session(export)
            second = compare_session(export)
        self.assertEqual(first, second)
        self.assertEqual(export, original)
        self.assertEqual(first["results"]["usd"]["net_pnl"], "0")
        momentum = first["results"]["demo_momentum"]
        self.assertEqual(momentum["entries"], 1)
        self.assertEqual(momentum["exits"], 1)
        self.assertGreater(D(momentum["fees"]), 0)
        self.assertGreater(D(momentum["turnover"]), 0)
        self.assertLess(D(momentum["net_pnl"]), 0)
        self.assertEqual(sum(p["count"] for p in first["temporal_split"]), 8)
        self.assertEqual(first["evidence"], "NO_VERIFIED_ADVANTAGE")

    def test_volatile_ids_dont_change_report(self):
        first = dataset()
        other = deepcopy(first)
        other["state"]["session_id"] = "other"
        for event in other["events"]:
            event["seq"] += 100
            if "quote_id" in event["payload"]:
                event["payload"]["quote_id"] = "other-" + event["payload"]["quote_id"]
        self.assertEqual(compare_session(first), compare_session(other))

    def test_valid_qwen_replay_uses_risk(self):
        export = dataset(qwen=True)
        add_decision(export)
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"], 1)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"], 1)
        self.assertLessEqual(D(report["results"]["recorded_qwen"]["turnover"]), D("510"))

    def test_answer_between_quotes_replays_only_explicit_fresh_recheck(self):
        export=dataset(qwen=True)
        _,answer=add_decision(export)
        when=stamp(NOW+timedelta(seconds=5,milliseconds=500))
        answer["occurred_at"]=answer["payload"]["processed_at"]=when
        quote=export["events"][5]["payload"]
        recheck=dict(kind="QUOTE_RECHECK",occurred_at=when,
                     payload=dict(quote_id=quote["quote_id"],observed_at=quote["observed_at"]))
        export["events"].append(recheck)
        export["events"].sort(key=lambda e:e["occurred_at"])
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],1)
        recheck["payload"]["quote_id"]="not-recorded"
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],0)

    def test_tampered_snapshot_never_replays(self):
        export = dataset(qwen=True)
        request, _ = add_decision(export)
        request["payload"]["snapshot"]["portfolio"]["cash"] = "999"
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["context_mismatches"], 1)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"], 0)

    def test_portfolio_mismatch_with_valid_hash_rejected(self):
        import hashlib
        export = dataset(qwen=True)
        request, answer = add_decision(export)
        snapshot = request["payload"]["snapshot"]
        snapshot["portfolio"]["cash"] = "999"
        snapshot["snapshot_id"] = hashlib.sha256(encode({k: v for k, v in snapshot.items() if k != "snapshot_id"}).encode()).hexdigest()
        answer["payload"]["snapshot_id"] = snapshot["snapshot_id"]
        answer["payload"]["answer"]["snapshot_id"] = snapshot["snapshot_id"]
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["context_mismatches"], 1)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"], 0)

    def test_future_request_is_not_visible_to_answer(self):
        export = dataset(qwen=True)
        add_decision(export, request_index=7, answer_index=6)
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["context_mismatches"], 1)
        self.assertEqual(report["results"]["recorded_qwen"]["entries"], 0)

    def test_stale_origin_snapshot_rejected(self):
        export = dataset(qwen=True)
        _, answer = add_decision(export)
        later = NOW + timedelta(seconds=40)
        answer["occurred_at"] = answer["payload"]["processed_at"] = stamp(later)
        raw = deepcopy(export["events"][-1]["payload"])
        raw.update(quote_id="late", observed_at=stamp(later))
        export["events"].append(dict(kind="SNAPSHOT", occurred_at=stamp(later), payload=raw))
        export["events"].sort(key=lambda e: e["occurred_at"])
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["context_mismatches"], 1)

    def test_duplicate_answer_not_executed_twice(self):
        export = dataset(qwen=True)
        _, answer = add_decision(export)
        duplicate = deepcopy(answer)
        duplicate["occurred_at"] = duplicate["payload"]["processed_at"] = export["events"][-1]["occurred_at"]
        export["events"].append(duplicate)
        report = compare_session(export)
        self.assertEqual(report["qwen_replay"]["replayed_decisions"], 1)
        self.assertEqual(report["qwen_replay"]["context_mismatches"], 1)

    def test_pause_and_user_stop_apply_to_all_engines(self):
        export = dataset()
        at = export["events"][6]["occurred_at"]
        export["events"].append(dict(kind="SESSION_STATUS", occurred_at=at,
                                     payload=dict(to="PAUSED_ENTRIES", reason=None)))
        export["events"].sort(key=lambda e: e["occurred_at"])
        self.assertEqual(compare_session(export)["results"]["demo_momentum"]["entries"], 0)
        export = dataset()
        export["events"].append(dict(kind="SESSION_STATUS", occurred_at=export["events"][-1]["occurred_at"],
                                     payload=dict(to="CLOSED", reason="USER_STOP")))
        report = compare_session(export)
        self.assertTrue(all(row["status"] == "CLOSED" for row in report["results"].values()))

    def test_watchdog_control_stops_all_arms_with_original_reason(self):
        for kind in ("WATCHDOG_TRIPPED", "SESSION_STATUS"):
            export = dataset()
            export["events"].append(dict(kind=kind, occurred_at=export["events"][-1]["occurred_at"],
                                         payload=dict(to="CLOSED", reason="SYSTEM_RESUME_GAP")))
            report = compare_session(export)
            self.assertTrue(all(row["status"] == "CLOSED" and row["exit_reason"] == "SYSTEM_RESUME_GAP"
                                for row in report["results"].values()))

    def test_wrong_source_future_and_backward_quotes_rejected(self):
        for field, value in (("source", "synthetic"), ("observed_at", stamp(NOW + timedelta(hours=1)))):
            export = dataset()
            export["events"][1]["payload"][field] = value
            with self.assertRaises(ValueError):
                compare_session(export)
        export = dataset()
        export["events"][1], export["events"][2] = export["events"][2], export["events"][1]
        with self.assertRaises(ValueError):
            compare_session(export)
        export = dataset()
        export["events"][2]["payload"]["observed_at"] = export["events"][1]["payload"]["observed_at"]
        with self.assertRaises(ValueError):
            compare_session(export)

    def test_risk_denial_counts_without_bypassing_limits(self):
        export = dataset()
        export["config"]["min_notional"] = "5000"
        report = compare_session(export)
        self.assertEqual(report["results"]["demo_momentum"]["entries"], 0)
        self.assertEqual(report["results"]["demo_momentum"]["denied_risk_count"], 1)

    def test_alpaca_replayed_with_local_model(self):
        export = dataset()
        export["config"]["source"] = "alpaca-paper"
        for event in export["events"]:
            if event["kind"] == "SNAPSHOT":
                event["payload"]["source"] = "alpaca-paper"
        report = compare_session(export)
        self.assertEqual(report["mode"], "OFFLINE_SIMULATION")
        self.assertGreater(report["results"]["demo_momentum"]["entries"], 0)


if __name__ == "__main__":
    unittest.main()
