"""Regression proof for submit-time AI expiry and quote-independent holding limits."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traid_ai.broker import BrokerEngine
from traid_ai.engine import Engine
from traid_ai.models import Config, D, Quote, parse_time, stamp, utc_now
from traid_ai.qwen import QwenStrategy
from traid_ai.store import Store
from test_alpaca import await_condition
from test_broker import EntryRule, FakeBroker
from test_qwen import ControlledModel


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "guards.sqlite3")
        self.now = utc_now()
        self.client = FakeBroker()
        self.engine = None

    def tearDown(self):
        if self.engine:
            if hasattr(self.engine.strategy, "close"):
                self.engine.strategy.close()
            if isinstance(self.engine, BrokerEngine):
                self.engine.close()
        self.store.close()
        self.tmp.cleanup()

    def quote(self, *, age=0):
        return Quote(stamp(self.now), stamp(self.now - timedelta(seconds=age)),
                     D("99.99"), D("100"), self.engine.config.source)

    def start(self, *, qwen=False, local=False):
        config = Config(source="synthetic" if local else "alpaca-paper", max_hold_seconds=10)
        if qwen:
            config = replace(config, strategy_version="qwen-local-v1", llm_digest="a" * 64,
                             decision_policy="legacy")
        if local:
            self.engine = Engine.start(self.store, config, self.now, self.now + timedelta(hours=2))
        else:
            self.engine = BrokerEngine.start_paper(self.store, config, self.now,
                                                  self.now + timedelta(hours=2), self.client)
        if qwen:
            self.engine.strategy.close()
            self.engine.strategy = QwenStrategy(client=ControlledModel())
            self.engine.state.history = ["100"] * 5
            self.store.commit_step(self.engine.state, "warmup", stamp(self.now), [])
            self.engine.step(self.now, self.quote())
            await_condition(self.engine.strategy.has_ready_result)
            self.now += timedelta(seconds=29)
        else:
            self.engine.strategy = EntryRule()

    def aborted_reason(self):
        return [e["payload"]["reason"] for e in self.store.events(self.engine.state.session_id)
                if e["kind"] == "ORDER_ABORTED"]

    def delay_commit(self, kind):
        clock = [self.now]
        original = self.store.commit_step
        def commit(state, step_id, at, events):
            result = original(state, step_id, at, events)
            if any(k == kind for k, p in events):
                clock[0] += timedelta(seconds=20)
            return result
        with patch.object(self.store, "commit_step", side_effect=commit), \
                patch("traid_ai.broker.utc_now", side_effect=lambda: clock[0]):
            self.engine.step(self.now, self.quote())
        self.assertEqual(self.client.posts, [])
        self.assertIsNone(self.engine.state.broker["pending"])
        self.assertEqual(self.aborted_reason(), ["AI_SNAPSHOT_EXPIRED_BEFORE_SUBMIT"])

    def test_ai_snapshot_expires_during_prepared_commit(self):
        self.start(qwen=True)
        self.delay_commit("ORDER_PREPARED")

    def test_ai_snapshot_expires_during_submitting_commit(self):
        self.start(qwen=True)
        self.delay_commit("ORDER_SUBMITTING")
        self.assertTrue(any(e["kind"] == "AI_DECISION" and e["payload"]["accepted"]
                            for e in self.store.events(self.engine.state.session_id)))

    def prepare_ai_entry(self):
        self.start(qwen=True)
        with patch.object(self.engine, "_dispatch"):
            self.engine.step(self.now, self.quote())
        return self.engine.state.broker["pending"]

    def restart(self):
        sid = self.engine.state.session_id
        self.engine.strategy.close()
        self.engine.close()
        self.engine = BrokerEngine(self.store, sid, self.client)

    def test_recovered_prepared_entry_keeps_original_basis(self):
        pending = self.prepare_ai_entry()
        basis = pending["ai_basis"]
        self.assertEqual(basis["snapshot_id"], self.engine.state.ai["snapshot_id"])
        self.assertEqual(basis["observed_at"], self.engine.state.ai["observed_at"])
        # New diagnostic state cannot renew the durable order's original basis.
        self.engine.state.ai = dict(snapshot_id="replacement", observed_at=stamp(self.now))
        self.store.commit_step(self.engine.state, "replace-ai-view", stamp(self.now), [])
        self.restart()
        self.now += timedelta(seconds=2)
        self.engine.step(self.now, None)
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.aborted_reason(), ["AI_SNAPSHOT_EXPIRED_BEFORE_SUBMIT"])

    def test_old_prepared_qwen_entry_without_basis_is_aborted(self):
        pending = self.prepare_ai_entry()
        pending.pop("ai_basis", None)
        self.store.commit_step(self.engine.state, "old-prepared", stamp(self.now), [])
        self.restart()
        self.engine.step(self.now, None)
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.aborted_reason(), ["AI_BASIS_MISSING_BEFORE_SUBMIT"])

    def test_fresh_recovered_prepared_entry_is_sent_once_at_age_limit(self):
        self.prepare_ai_entry()
        self.restart()
        self.now += timedelta(seconds=1)
        with patch("traid_ai.broker.utc_now", return_value=self.now):
            self.engine.step(self.now, None)
            self.now += timedelta(seconds=2)
            self.engine.step(self.now, None)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy"])
        self.assertEqual(self.aborted_reason(), [])

    def test_expired_basis_does_not_abort_unknown_already_sent_order(self):
        self.start(qwen=True)
        self.client.timeout_after_submit = True
        self.engine.step(self.now, self.quote())
        self.now += timedelta(seconds=3)
        self.engine.step(self.now, None)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy"])
        self.assertIsNotNone(self.engine.state.broker["pending"])
        self.assertEqual(self.aborted_reason(), [])

    def enter_position(self, *, local=False, partial=False):
        self.start(local=local)
        self.client.auto_fill = not partial
        self.engine.step(self.now, self.quote())
        if not local:
            if partial:
                p = self.engine.state.broker["pending"]
                self.client.fill(p["request"]["client_order_id"], D(p["request"]["qty"]) / 2)
            self.now += timedelta(seconds=1)
            self.engine.step(self.now, None)
        self.now = parse_time(self.engine.state.entered_at) + timedelta(seconds=10)

    def test_max_hold_sends_reduce_only_exit_without_new_quote(self):
        self.enter_position()
        self.engine.step(self.now, None)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy", "sell"])
        self.assertEqual(self.engine.state.broker["last_order"]["side"], "sell")
        self.assertEqual(self.engine.state.position_exit_reason, "MAX_HOLD")

    def test_max_hold_ignores_rejected_market_price(self):
        self.enter_position()
        events = self.engine.step(self.now, self.quote(age=60))
        self.assertTrue(any(k == "DATA_REJECTED" for k, p in events))
        self.assertEqual([p["side"] for p in self.client.posts], ["buy", "sell"])
        self.assertEqual(self.engine.state.position_exit_reason, "MAX_HOLD")

    def test_max_hold_latches_while_reconciliation_is_pending(self):
        self.enter_position()
        with patch.object(self.engine.read_cycle, "poll", return_value=None):
            self.engine.step(self.now, None)
        self.assertEqual(self.engine.state.position_exit_reason, "MAX_HOLD")
        self.assertEqual([p["side"] for p in self.client.posts], ["buy"])
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy", "sell"])

    def test_partial_buy_is_canceled_and_reconciled_before_max_hold_sale(self):
        self.enter_position(partial=True)
        self.client.cancel_race = True
        self.engine.step(self.now, None)
        self.assertEqual(len(self.client.cancels), 1)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy"])
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual([p["side"] for p in self.client.posts], ["buy", "sell"])
        self.assertEqual(D(self.client.posts[-1]["qty"]), self.client.qty)
        self.assertEqual(self.engine.state.broker["pending"]["reason"], "MAX_HOLD")

    def test_local_max_hold_waits_for_valid_quote_and_survives_restart(self):
        self.enter_position(local=True)
        qty = self.engine.state.quantity
        events = self.engine.step(self.now, None)
        self.assertEqual(self.engine.state.position_exit_reason, "MAX_HOLD")
        self.assertEqual(self.engine.state.quantity, qty)
        self.assertFalse(any(k == "PAPER_FILL" for k, p in events))
        self.engine = Engine(self.store, self.engine.state.session_id)
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, self.quote(age=60))
        self.assertEqual(self.engine.state.quantity, qty)
        self.now += timedelta(seconds=1)
        events = self.engine.step(self.now, self.quote())
        self.assertTrue(any(k == "PAPER_FILL" and p["reason"] == "MAX_HOLD" for k, p in events))
        self.assertEqual(D(self.engine.state.quantity), 0)
