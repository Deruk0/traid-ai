from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from traid_ai.analytics import (MAX_SAMPLES, MEMORY_EVENTS, analyze, remember_quote,
                                remember_events, session_memory)
from traid_ai.engine import Engine
from traid_ai.models import Config, D, Quote, State, encode, stamp, utc_now
from traid_ai.qwen import build_snapshot
from traid_ai.store import Store


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.now = utc_now()
        self.config = Config(source="kraken-public", strategy_version="qwen-local-v1", llm_digest="a"*64)
        self.state = State("session", stamp(self.now), stamp(self.now+timedelta(hours=1)),
                           "RUNNING", "1000", "1000")

    def quote(self, seconds, price="100", source="kraken-public"):
        return Quote(str(seconds), stamp(self.now+timedelta(seconds=seconds)), D(price), D(price), source)

    def sample(self, seconds, price="100"):
        quote = self.quote(seconds, price)
        remember_quote(self.state, quote, self.now+timedelta(seconds=seconds))
        return quote

    def test_complete_horizons_have_measured_returns_and_estimated_costs(self):
        for t in range(0, 901, 10):
            q = self.sample(t, str(D("100")+D(t)/900))
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=900))
        self.assertEqual([w["status"] for w in result["windows"]], ["READY"]*3)
        self.assertEqual(result["windows"][2]["return_pct"], "1.000000")
        self.assertGreater(D(result["windows"][2]["sampled_variation_pct"]), 0)
        self.assertEqual(result["costs"]["estimated_round_trip_bps"], "30.000000")
        self.assertTrue(result["costs"]["is_estimate"])
        self.assertFalse(result["quality"]["volume_available"])
        self.assertFalse(result["quality"]["candle_history_available"])

    def test_short_history_and_gaps_cannot_claim_complete_horizon(self):
        self.sample(0)
        q = self.sample(10, "101")
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=10))
        self.assertEqual(result["windows"][0]["status"], "WARMUP")
        self.assertIsNone(result["windows"][0]["return_pct"])
        self.assertEqual(result["windows"][0]["observed_change_pct"], "1.000000")
        q = self.sample(60, "102")
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=60))
        window = result["windows"][0]
        self.assertEqual(window["status"], "GAPS")
        self.assertEqual(window["gap_count"], 1)
        self.assertIsNone(window["return_pct"])
        self.assertIsNone(window["sampled_variation_pct"])
        for t in range(70, 131, 10):
            q = self.sample(t)
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=130))
        self.assertEqual(result["windows"][0]["status"], "READY")
        self.assertEqual(result["windows"][1]["status"], "GAPS")

    def test_conflicting_horizons_are_reported_without_predicting(self):
        for t in range(0, 301, 10):
            q = self.sample(t, "100" if t < 240 else "102" if t == 240 else "101")
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=300))
        self.assertTrue(result["quality"]["conflicting_horizons"])
        self.assertFalse(result["reference_signal"]["validated_trading_edge"])

    def test_future_unavailable_and_other_source_samples_are_excluded(self):
        self.sample(0)
        q = self.sample(10)
        hidden = deepcopy(self.state.quote_samples[0])
        hidden.update(quote_id="unavailable", mid="999", available_at=stamp(self.now+timedelta(days=1)))
        self.state.quote_samples.insert(0, hidden)
        foreign = dict(hidden, source="alpaca-paper", available_at=stamp(self.now))
        self.state.quote_samples.insert(0, foreign)
        self.sample(20, "999")
        result = analyze(self.config, self.state, q, self.now+timedelta(seconds=10))
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["windows"][0]["observed_change_pct"], "0.000000")

    def test_history_is_bounded_by_count_and_time(self):
        for t in range(MAX_SAMPLES+15):
            self.sample(t)
        self.assertEqual(len(self.state.quote_samples), MAX_SAMPLES)
        self.sample(5000)
        self.assertEqual(len(self.state.quote_samples), 1)

    def test_snapshot_changes_memory_provenance_and_hash_survive_restart(self):
        self.state.history = ["100"]*5
        q = self.sample(0)
        snap = build_snapshot(self.config, self.state, q, self.now)
        self.assertIsNone(snap["changes"])
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"state.sqlite3")
            store.create(self.config, self.state)
            events = [("AI_REQUEST", {"snapshot": snap}),
                      ("AI_DECISION", dict(snapshot_id=snap["snapshot_id"], action="REQUEST_ENTRY", accepted=True,
                                           reason="DO_NOT_REPLAY_THIS_MODEL_TEXT")),
                      ("RISK_VERDICT", dict(approved=False, reason="BUDGET_OR_MINIMUM")),
                      ("PAPER_FILL", dict(side="BUY", quantity="1", price="100", account_id="SECRET")),
                      ("BROKER_UNRESOLVED", dict(reason="SECRET_BROKER_ERROR", account_id="SECRET"))]
            store.commit_step(self.state, "one", stamp(self.now), events)
            store.close()
            store = Store(Path(tmp)/"state.sqlite3")
            cfg, state = store.load("session")
            self.assertEqual(state.quote_samples, self.state.quote_samples)
            state.broker = {"account_id": "SECRET", "pending": None}
            quote = self.quote(10, "101")
            remember_quote(state, quote, self.now+timedelta(seconds=10))
            rebuilt = build_snapshot(cfg, state, quote, self.now+timedelta(seconds=10))
            self.assertEqual(rebuilt["snapshot_version"], "market-summary-v7")
            self.assertEqual(rebuilt["changes"]["mid_change_pct"], "1.000000")
            self.assertEqual(rebuilt["changes"]["previous_snapshot_id"], snap["snapshot_id"])
            self.assertTrue(rebuilt["memory"]["last_decision"]["accepted"])
            self.assertEqual(rebuilt["memory"]["recent_events"][0]["reason_code"], "BUDGET_OR_MINIMUM")
            serialized = encode(rebuilt)
            self.assertNotIn("SECRET", serialized)
            self.assertNotIn("DO_NOT_REPLAY", serialized)
            copy = dict(rebuilt)
            digest = copy.pop("snapshot_id")
            self.assertEqual(digest, hashlib.sha256(encode(copy).encode()).hexdigest())
            saved = store.events("session")[1]["payload"]["snapshot"]
            self.assertEqual(saved, json.loads(encode(snap)))
            store.close()

    def test_memory_is_bounded_and_failed_commit_leaves_no_durable_fact(self):
        for t in range(20):
            remember_events(self.state, stamp(self.now+timedelta(seconds=t)), [
                ("PAPER_FILL", dict(side="SELL", quantity="1", price="100"))])
        self.assertEqual(len(session_memory(self.state, self.now+timedelta(seconds=20))["recent_events"]), MEMORY_EVENTS)
        self.assertEqual(session_memory(self.state, self.now)["recent_events"], [])
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"state.sqlite3")
            store.create(self.config, self.state)
            before = store.load("session")[1].context_memory
            # Circular JSON forces failure after memory projection, before commit.
            with self.assertRaises(ValueError):
                self._failed_commit(store)
            self.assertEqual(store.load("session")[1].context_memory, before)
            self.assertFalse(store.seen("session", "fail"))
            store.close()

    def _failed_commit(self, store):
        circular = {}
        circular["self"] = circular
        store.commit_step(self.state, "fail", stamp(self.now), [
            ("SESSION_STATUS", dict(to="PAUSED_ENTRIES")), ("INVALID", circular)])

    def test_engine_rejects_quote_without_polluting_analytics_and_old_state_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"state.sqlite3")
            old = self.state.to_dict()
            old.pop("quote_samples")
            old.pop("context_memory")
            store.create(self.config, self.state)
            with store.db:
                store.db.execute("UPDATE sessions SET state=? WHERE id=?", (encode(old), "session"))
            engine = Engine(store, "session")
            self.assertEqual(engine.state.quote_samples, [])
            self.assertEqual(engine.state.context_memory, {})
            engine.step(self.now, self.quote(0), step_id="first")
            engine.step(self.now+timedelta(seconds=1), self.quote(1, source="alpaca-paper"), step_id="wrong")
            engine.step(self.now+timedelta(seconds=2), self.quote(0), step_id="duplicate-observed")
            self.assertEqual(len(engine.state.quote_samples), 1)
            self.assertEqual(engine.state.quote_samples[0]["quote_id"], "0")
            store.close()


if __name__ == "__main__":
    unittest.main()
