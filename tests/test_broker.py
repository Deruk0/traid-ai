from copy import deepcopy
from datetime import timedelta
from decimal import ROUND_DOWN
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traid_ai.alpaca import AlpacaError, PaperConnection
from traid_ai.broker import BrokerEngine, account_lock_path
from traid_ai.dashboard import SessionManager, chart_data
from traid_ai.models import Config, D, Quote, stamp, utc_now
from traid_ai.store import Store, writer_lock
from traid_ai.strategy import Intent
from test_alpaca import await_condition


class EntryRule:
    version = "test"
    def decide(self, history, mid, position):
        return Intent("WAIT" if position else "REQUEST_ENTRY", "TEST_ENTRY")


class FakeBroker:
    def __init__(self):
        self.account_id = "test-paper-broker-account"
        self.cash, self.qty = D("100000"), D("0")
        self.records = {}
        self.posts = []
        self.cancels = []
        self.auto_fill = False
        self.timeout_after_submit = False
        self.timeout_before_submit = False
        self.cancel_race = False
        self.foreign = False
        self.before_post = None
        self.ask = D("100")

    def account(self):
        return dict(account_id=self.account_id, account_suffix="test00", status="ACTIVE",
                    cash=str(self.cash), equity=str(self.cash + self.qty * 100),
                    buying_power=str(self.cash * 4), non_marginable_buying_power=str(self.cash),
                    account_blocked=False, trading_blocked=False)
    def asset(self):
        return dict(min_order_size="0.00000001", min_trade_increment="0.00000001", price_increment="0.01")
    def positions(self):
        return [dict(symbol="BTCUSD", qty=str(self.qty))] if self.qty else []
    def open_orders(self):
        rows = [deepcopy(r) for r in self.records.values() if r["status"] not in
                {"filled", "canceled", "expired", "rejected"}]
        if self.foreign:
            rows.append(dict(id="external", client_order_id="manual", symbol="BTCUSD"))
        return rows
    def orders(self):
        return [deepcopy(r) for r in self.records.values()]
    def quote(self):
        return dict(symbol="BTC/USD", bid=str(self.ask - D(".01")), ask=str(self.ask),
                    observed_at=stamp(utc_now()), stale=False)
    def submit_order(self, req):
        if self.before_post:
            self.before_post(req)
        self.posts.append(deepcopy(req))
        if self.timeout_before_submit:
            raise AlpacaError("offline")
        r = dict(id="order-" + str(len(self.posts)), **deepcopy(req), status="new",
                 filled_qty="0", filled_avg_price=None)
        self.records[req["client_order_id"]] = r
        if self.auto_fill:
            self.fill(req["client_order_id"], D(req["qty"]))
        if self.timeout_after_submit:
            self.timeout_after_submit = False
            raise AlpacaError("reply lost")
        return deepcopy(r)
    def order_by_client_id(self, cid):
        return deepcopy(self.records.get(cid))
    def fill(self, cid, delta):
        r = self.records[cid]
        price = D(r.get("limit_price") or self.ask)
        if r["side"] == "buy":
            self.cash -= delta * price
            net = (delta * D(".9975")).quantize(D(".00000001"), rounding=ROUND_DOWN)
            self.qty += net
        else:
            self.qty -= delta
            self.cash += delta * price * D(".9975")
        r["filled_qty"] = str(D(r["filled_qty"]) + delta)
        r["filled_avg_price"] = str(price)
        r["status"] = "filled" if D(r["filled_qty"]) == D(r["qty"]) else "partially_filled"
    def cancel_order(self, oid):
        self.cancels.append(oid)
        r = next(r for r in self.records.values() if r["id"] == oid)
        if self.cancel_race:
            self.fill(r["client_order_id"], D(r["qty"]) - D(r["filled_qty"]))
            self.cancel_race = False
        else:
            r["status"] = "canceled"


class BrokerTests(unittest.TestCase):
    def test_stop_during_submitting_commit_cannot_send_buy(self):
        control=[False,None]
        self.engine.control_state=lambda:tuple(control)
        original=self.engine._save
        def saved(now,events):
            original(now,events)
            if any(kind=="ORDER_SUBMITTING" for kind,payload in events):
                control[0]=True
        self.engine._save=saved
        self.tick()
        self.assertEqual(self.client.posts,[])
        self.assertIsNone(self.engine.state.broker["pending"])
        self.assertEqual(self.engine.state.status,"FLATTENING")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sessions.sqlite3"
        self.store = Store(self.path)
        self.client = FakeBroker()
        self.now = utc_now()
        self.engine = BrokerEngine.start_paper(self.store, Config(source="alpaca-paper"),
                                              self.now, self.now + timedelta(hours=1), self.client)
        self.engine.strategy = EntryRule()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def tick(self, **kwargs):
        self.now += timedelta(seconds=1)
        q = Quote(stamp(self.now), stamp(self.now), D("99.99"), D("100"), "alpaca-paper")
        return self.engine.step(self.now, q, **kwargs)

    def pending(self):
        return self.engine.state.broker["pending"]

    def test_persist_before_send_no_invented_fill_and_limit_risk(self):
        def check(req):
            config, state = self.store.load(self.engine.state.session_id)
            self.assertEqual(state.broker["pending"]["request"], req)
            self.assertTrue(state.broker["pending"]["attempted"])
            self.assertEqual(D(state.quantity), 0)
        self.client.before_post = check
        self.tick()
        req = self.client.posts[0]
        self.assertEqual((req["type"], req["time_in_force"]), ("limit", "ioc"))
        qty, price = D(req["qty"]), D(req["limit_price"])
        self.assertLessEqual(qty * price * D("1.0025"), D("250"))
        self.assertEqual(D(self.engine.state.quantity), 0)
        self.assertEqual(self.engine.state.entries, 0)
        self.assertTrue(self.pending())

    def test_partial_fills_are_monotonic_and_ledger_matches_real_position(self):
        self.tick()
        cid, qty = self.pending()["request"]["client_order_id"], D(self.pending()["request"]["qty"])
        self.client.fill(cid, qty / 2)
        self.tick()
        expected_qty = self.client.qty
        fees = self.engine.state.fees
        self.tick()
        self.assertEqual(self.engine.state.fees, fees)
        self.assertEqual(D(self.engine.state.quantity), expected_qty)
        self.assertEqual(self.engine.state.entries, 1)
        self.client.fill(cid, qty / 2)
        self.tick()
        self.assertIsNone(self.pending())
        self.assertEqual(D(self.engine.state.quantity), self.client.qty)
        self.assertEqual(D(self.engine.state.cash), D("1000") + self.client.cash - D("100000"))
        self.assertEqual(self.engine.state.entries, 1)

    def test_timeout_with_accepted_order_recovers_and_never_rebuys(self):
        self.client.auto_fill = self.client.timeout_after_submit = True
        self.tick()
        self.assertEqual(self.engine.state.status, "UNRESOLVED")
        session_id = self.engine.state.session_id
        self.engine = BrokerEngine(self.store, session_id, self.client)
        self.tick()
        self.tick()
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(sum(r["side"] == "buy" for r in self.client.posts), 1)
        self.assertEqual(self.client.qty, 0)

    def test_unknown_absent_order_cannot_be_blindly_retried(self):
        self.client.timeout_before_submit = True
        self.tick()
        for _ in range(4):
            self.tick(stop=True)
        self.assertEqual(len(self.client.posts), 1)
        self.assertEqual(self.engine.state.status, "UNRESOLVED")
        self.assertIsNotNone(self.pending())

    def test_cancel_fill_race_is_reconciled_before_exit(self):
        self.tick()
        self.client.cancel_race = True
        self.tick(stop=True)
        self.assertNotEqual(self.engine.state.status, "CLOSED")
        self.tick(stop=True)
        self.assertEqual(self.client.posts[-1]["side"], "sell")
        self.client.fill(self.pending()["request"]["client_order_id"], D(self.pending()["request"]["qty"]))
        self.tick(stop=True)
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(self.client.qty, 0)
        self.assertEqual(sum(r["side"] == "buy" for r in self.client.posts), 1)

    def test_missing_order_requires_repeated_full_reconciliation_then_closes(self):
        self.client.timeout_before_submit = True
        self.tick()
        for delay in (10, 10, 40):
            self.now += timedelta(seconds=delay)
            self.tick(stop=True)
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertIsNone(self.pending())
        self.assertEqual(len(self.client.posts), 1)
        self.assertEqual(self.engine.state.broker["last_order"]["status"],
                         "not_found_after_reconciliation")

    def test_history_or_account_change_prevents_missing_order_confirmation(self):
        self.client.timeout_after_submit = True
        self.tick()
        cid = self.pending()["request"]["client_order_id"]
        self.client.order_by_client_id = lambda _: None
        for _ in range(3):
            self.now += timedelta(seconds=30)
            self.tick(stop=True)
        self.assertEqual(self.engine.state.status, "UNRESOLVED")
        self.assertIsNotNone(self.pending())
        self.client.records.pop(cid)
        self.client.cash -= D("10")
        for _ in range(3):
            self.now += timedelta(seconds=30)
            self.tick(stop=True)
        self.assertIsNotNone(self.pending())
        self.assertEqual(len(self.client.posts), 1)

    def test_quote_expiring_before_submit_aborts_entry(self):
        later = self.now + timedelta(minutes=5)
        with patch("traid_ai.broker.utc_now", return_value=later):
            self.tick()
        self.assertEqual(self.client.posts, [])
        self.assertIsNone(self.pending())

    def test_pause_cancels_unfilled_entry_without_new_order(self):
        self.tick()
        self.tick(pause_entries=True)
        self.tick()
        self.assertEqual(self.engine.state.status, "PAUSED_ENTRIES")
        self.assertEqual(len(self.client.posts), 1)
        self.assertEqual(self.client.qty, 0)

    def test_deadline_exit_does_not_need_fake_market_fill(self):
        self.client.auto_fill = True
        self.tick()
        self.tick()
        deadline = self.engine.state.deadline_at
        self.now += timedelta(hours=1)
        self.engine.step(self.now, None, error="market feed offline")
        self.assertEqual(self.client.posts[-1]["side"], "sell")
        self.assertNotEqual(self.engine.state.status, "CLOSED")
        self.now += timedelta(seconds=1)
        self.engine.step(self.now, None)
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(self.engine.state.deadline_at, deadline)

    def test_foreign_order_and_wrong_account_block_mutations(self):
        self.client.foreign = True
        self.tick()
        self.assertEqual(self.engine.state.status, "UNRESOLVED")
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.client.cancels, [])
        self.client.foreign = False
        self.client.account_id = "different-account"
        self.tick()
        self.assertEqual(self.client.posts, [])

    def test_restart_partial_and_stop_preserve_budget_and_close_actual_net_qty(self):
        self.tick()
        cid, qty = self.pending()["request"]["client_order_id"], D(self.pending()["request"]["qty"])
        self.client.fill(cid, qty / 2)
        self.tick()
        session_id, deadline = self.engine.state.session_id, self.engine.state.deadline_at
        self.engine = BrokerEngine(self.store, session_id, self.client)
        self.tick(stop=True)
        self.client.auto_fill = True
        self.tick(stop=True)
        self.tick(stop=True)
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(D(self.engine.state.quantity), 0)
        self.assertEqual(self.client.qty, 0)
        self.assertEqual(self.engine.state.deadline_at, deadline)
        self.assertEqual(self.engine.config.capital, D("1000"))
        data = chart_data(self.engine.config, self.engine.state, self.store.events(session_id))
        self.assertEqual(D(data["points"][-1]["equity"]), D(self.engine.report()["equity_at_last_quote"]))
        self.assertTrue(self.engine.report()["fees_are_estimated"])

    def test_fail_to_persist_never_submits(self):
        original = self.store.commit_step
        def fail_prepared(state, step_id, now, events):
            if any(kind == "ORDER_PREPARED" for kind, _ in events):
                raise OSError("disk unavailable")
            return original(state, step_id, now, events)
        self.store.commit_step = fail_prepared
        with self.assertRaises(OSError):
            self.tick()
        self.assertEqual(self.client.posts, [])

    def test_new_session_requires_clean_account_and_cash_without_margin(self):
        self.client.qty = D("1")
        with self.assertRaises(ValueError):
            BrokerEngine.start_paper(self.store, self.engine.config, self.now,
                                    self.now + timedelta(hours=1), self.client)
        self.client.qty, self.client.cash = D("0"), D("999")
        with self.assertRaises(ValueError):
            BrokerEngine.start_paper(self.store, self.engine.config, self.now,
                                    self.now + timedelta(hours=1), self.client)

    def test_account_lock_is_independent_of_database_path(self):
        first = account_lock_path("paper-one")
        self.assertEqual(first, account_lock_path("paper-one"))
        self.assertNotEqual(first, account_lock_path("paper-two"))
        with writer_lock(first):
            with self.assertRaises(ValueError):
                with writer_lock(account_lock_path("paper-one")):
                    self.fail("second owner")

    def test_smoke_cycle_runs_once_and_closes(self):
        now = utc_now()
        engine = BrokerEngine.start_paper(self.store, Config(source="alpaca-paper",
              capital=D("100"), max_exposure=D(".25"), strategy_version="paper-smoke-v1"),
              now, now + timedelta(minutes=2), self.client)
        self.client.auto_fill = True
        for n in range(3):
            when = now + timedelta(seconds=n + 1)
            engine.step(when, Quote(str(n), stamp(when), D("99.99"), D("100"), "alpaca-paper"))
        self.assertEqual(engine.state.status, "CLOSED")
        self.assertEqual([r["side"] for r in self.client.posts], ["buy", "sell"])
        self.assertEqual(self.client.qty, 0)

    def test_clock_crossing_deadline_between_plan_and_submit_aborts_entry(self):
        later = self.now + timedelta(hours=2)
        with patch("traid_ai.broker.utc_now", return_value=later):
            self.tick()
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.engine.state.status, "FLATTENING")

    def test_minimum_order_does_not_override_small_exposure_budget(self):
        engine = BrokerEngine.start_paper(self.store, Config(source="alpaca-paper",
                capital=D("100"), max_exposure=D(".1"), strategy_version="paper-smoke-v1"),
                self.now, self.now + timedelta(hours=1), self.client)
        when = self.now + timedelta(seconds=1)
        engine.step(when, Quote("small", stamp(when), D("99.99"), D("100"), "alpaca-paper"))
        self.assertEqual(self.client.posts, [])
        self.assertEqual(engine.config.min_notional, D("10"))

    def test_stop_or_pause_arriving_during_reconciliation_prevents_post(self):
        for stop, pause in [(True, None), (False, True)]:
            with self.subTest(stop=stop):
                engine = BrokerEngine.start_paper(self.store, Config(source="alpaca-paper"),
                    self.now, self.now + timedelta(hours=1), self.client)
                engine.strategy = EntryRule()
                control = [False, None]
                engine.control_state = lambda: tuple(control)
                original = self.client.account
                def account():
                    control[:] = [stop, pause]
                    return original()
                self.client.account = account
                when = self.now + timedelta(seconds=1)
                engine.step(when, Quote("race", stamp(when), D("99.99"), D("100"), "alpaca-paper"))
                self.client.account = original
                self.assertEqual(self.client.posts, [])
                self.assertIsNone(engine.state.broker["pending"])
                self.assertEqual(engine.state.status, "FLATTENING" if stop else "PAUSED_ENTRIES")


class BrokerManagerTests(unittest.TestCase):
    def test_dashboard_launch_pause_stop_and_account_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(Path(directory) / "paper.sqlite3")
            client = FakeBroker()
            manager.alpaca = PaperConnection(client_factory=lambda k,s: client, poll_seconds=100)
            try:
                manager.alpaca.connect("key", "secret")
                sid = manager.launch(payload={"mode":"alpaca", "duration_minutes":"2"})
                await_condition(lambda: manager.view()["state"]["last_tick_at"] is not None)
                self.assertEqual(manager.view()["report"]["mode"], "ALPACA_PAPER")
                with self.assertRaises(ValueError):
                    manager.connection_change()
                manager.control("pause", sid)
                await_condition(lambda: manager.view()["report"]["status"] == "PAUSED_ENTRIES")
                manager.control("stop", sid)
                await_condition(lambda: manager.view()["report"]["status"] == "CLOSED")
                await_condition(lambda: not manager.running)
                self.assertEqual(client.posts, [])
            finally:
                manager.shutdown()

    def test_same_account_different_databases_cannot_have_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            first = SessionManager(Path(directory) / "one.sqlite3")
            second = SessionManager(Path(directory) / "two.sqlite3")
            client = FakeBroker()
            for manager in (first, second):
                manager.alpaca = PaperConnection(client_factory=lambda k,s: client, poll_seconds=100)
                manager.alpaca.connect("key", "secret")
            try:
                sid = first.launch(payload={"mode":"alpaca", "duration_minutes":"2"})
                with self.assertRaises(ValueError):
                    second.launch(payload={"mode":"alpaca", "duration_minutes":"2"})
                self.assertEqual(second.view()["sessions"], [])
                first.control("stop", sid)
                await_condition(lambda: not first.running)
                next_sid = second.launch(payload={"mode":"alpaca", "duration_minutes":"2"})
                second.control("stop", next_sid)
                await_condition(lambda: not second.running)
            finally:
                first.shutdown()
                second.shutdown()


if __name__ == "__main__":
    unittest.main()
