from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from traid_ai.accounting import reconcile_activities
from traid_ai.models import Config, D, State, stamp


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.day = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
        self.now = self.day+timedelta(days=1)
        self.config = Config(source="alpaca-paper")
        self.state = State("session", stamp(self.day-timedelta(minutes=1)), stamp(self.day+timedelta(hours=2)),
                           "CLOSED", "1008.4", "1008.4", broker={"account_id": "paper", "initial_cash": "100000", "pending": None})
        self.events = [self.event("buy", "BUY", "1", "100", 0),
                       self.event("sell", "SELL", ".99", "110", 60)]
        self.activities = [self.fill("buy-fill", "buy", "buy", "1", "100", 0),
                           self.fill("sell-fill", "sell", "sell", ".99", "110", 60),
                           dict(id="btc-fee", activity_type="CFEE", date="2026-10-14", net_amount="0",
                                symbol="BTCUSD", qty="-.01", price="100", status="executed"),
                           dict(id="usd-fee", activity_type="FEE", date="2026-10-14", net_amount="-.5")]
        self.account = dict(account_id="paper", currency="USD", cash="100008.4")

    def event(self, order, side, qty, price, seconds):
        return dict(kind="BROKER_FILL", occurred_at=stamp(self.day+timedelta(seconds=seconds)),
                    payload=dict(order_id=order, side=side, quantity=qty, price=price, fee="999"))

    def fill(self, activity_id, order, side, qty, price, seconds):
        return dict(id=activity_id, activity_type="FILL", order_id=order, side=side,
                    qty=qty, price=price, symbol="BTC/USD", transaction_time=stamp(self.day+timedelta(seconds=seconds)))

    def report(self, **kwargs):
        values = dict(config=self.config, state=self.state, events=self.events,
                      activities=self.activities, now=self.now, account=self.account,
                      positions=[], open_orders=[])
        values.update(kwargs)
        return reconcile_activities(**values)

    def test_late_btc_fee_is_valued_once_without_double_charging_cash(self):
        self.state.fees = "999"
        self.state.accounting = {"fees_usd": "888"}
        result = self.report()
        self.assertEqual(result["status"], "RECONCILED_AS_OF")
        self.assertEqual(D(result["fees_usd"]), D("1.5"))
        self.assertEqual(D(result["fees_by_asset"]["BTC"]), D(".01"))
        self.assertEqual(D(result["fees_by_asset"]["USD"]), D(".5"))
        self.assertEqual(D(result["net_pnl"]), D("8.4"))
        self.assertEqual(D(result["quantity"]), 0)
        self.assertTrue(result["pnl_is_final"])
        self.assertFalse(result["fees_are_estimated"])
        self.assertTrue(result["balances_match"])
        self.assertTrue(result["fills_match"])
        self.assertTrue(result["has_posted_fees"])

    def test_repeated_reads_and_identical_duplicate_ids_are_idempotent(self):
        before = deepcopy((self.state.to_dict(), self.events, self.activities))
        first = self.report()
        second = self.report(activities=self.activities+[deepcopy(self.activities[-1])])
        self.assertEqual(first, second)
        self.assertEqual(before, (self.state.to_dict(), self.events, self.activities))
        self.assertEqual(len(first["activity_ids"]), 4)

    def test_conflicting_duplicate_id_is_unresolved(self):
        duplicate = dict(self.activities[-1], net_amount="-1")
        result = self.report(activities=self.activities+[duplicate])
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIn("CONFLICTING_ACTIVITY_ID", result["reason_codes"])
        self.assertFalse(result["balances_match"])
        self.assertIsNone(result["net_pnl"])

    def test_day_aggregate_requires_exclusive_selected_trades(self):
        external = self.fill("external", "foreign", "buy", "1", "100", -30)
        result = self.report(activities=self.activities+[external])
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIn("UNALLOCATED_FEE", result["reason_codes"])
        self.assertEqual(result["unallocated_activity_ids"], ["btc-fee", "external", "usd-fee"])
        # Even earlier trades that were in initial_cash make daily fees ambiguous.
        external["transaction_time"] = stamp(self.day-timedelta(hours=1))
        result = self.report(activities=self.activities+[external], other_order_ids=("foreign",))
        self.assertIn("UNALLOCATED_FEE", result["reason_codes"])

    def test_exact_order_fees_do_not_claim_foreign_earlier_trades(self):
        activities = deepcopy(self.activities)
        activities[2]["order_id"] = "buy"
        activities[3]["order_id"] = "sell"
        activities.append(self.fill("earlier", "foreign", "buy", "1", "100", -3600))
        result = self.report(activities=activities)
        self.assertEqual(result["status"], "RECONCILED_AS_OF")

    def test_unknown_order_and_unmatched_fill_are_never_allocated(self):
        activities = deepcopy(self.activities)
        activities[-1]["order_id"] = "foreign"
        result = self.report(activities=activities)
        self.assertIn("UNALLOCATED_FEE", result["reason_codes"])
        self.assertIn("usd-fee", result["unallocated_activity_ids"])
        events = self.events+[dict(kind="ORDER_UPDATE", occurred_at=stamp(self.day),
                                  payload=dict(order_id="unused", side="buy", filled_qty="0"))]
        activities = self.activities+[self.fill("unconfirmed", "unused", "buy", "1", "100", 30)]
        result = self.report(events=events, activities=activities)
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_missing_fill_activity_is_pending(self):
        result = self.report(activities=self.activities[1:])
        self.assertEqual(result["status"], "PENDING_ACTIVITIES")
        self.assertIn("MISSING_FILL_ACTIVITIES", result["reason_codes"])
        self.assertFalse(result["fills_match"])
        self.assertFalse(result["balances_match"])

    def test_missing_fees_and_same_day_posting_stay_pending(self):
        events = [self.events[0], self.event("sell", "SELL", "1", "110", 60)]
        activities = [self.activities[0], self.fill("sell-fill", "sell", "sell", "1", "110", 60)]
        result = self.report(events=events, activities=activities,
                             account=dict(self.account, cash="100010"))
        self.assertEqual(result["status"], "PENDING_ACTIVITIES")
        self.assertIn("MISSING_FEE_ACTIVITIES", result["reason_codes"])
        self.assertFalse(result["has_posted_fees"])
        activities = deepcopy(self.activities)
        for activity in activities[2:]:
            activity["date"] = "2026-10-13"
        result = self.report(activities=activities, now=self.day+timedelta(hours=2))
        self.assertEqual(result["status"], "PENDING_ACTIVITIES")
        self.assertIn("END_OF_DAY_FEES_PENDING", result["reason_codes"])
        self.assertTrue(result["has_posted_fees"])
        self.assertFalse(result["pnl_is_final"])

    def test_active_session_can_confirm_posted_balance_but_not_final_pnl(self):
        self.state.status = "RUNNING"
        result = self.report()
        self.assertEqual(result["status"], "PENDING_ACTIVITIES")
        self.assertTrue(result["balances_match"])
        self.assertTrue(result["fills_match"])
        self.assertTrue(result["has_posted_fees"])
        self.assertIsNone(result["net_pnl"])

    def test_future_and_malformed_activities_fail_closed(self):
        for patch in (dict(transaction_time=stamp(self.now+timedelta(seconds=1))),
                      dict(qty="NaN"), dict(price="0"), dict(side="sell"), dict(symbol="ETHUSD")):
            with self.subTest(patch=patch):
                activities = deepcopy(self.activities)
                activities[0].update(patch)
                result = self.report(activities=activities)
                self.assertEqual(result["status"], "UNRESOLVED")
                self.assertFalse(result["balances_match"])
        result = self.report(activities=[None])
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_crypto_fee_missing_price_mixed_money_and_foreign_asset_rejected(self):
        for patch in (dict(price=""), dict(price="0"), dict(net_amount="-1"),
                      dict(symbol="ETHUSD"), dict(qty=".01")):
            with self.subTest(patch=patch):
                activities = deepcopy(self.activities)
                activities[2].update(patch)
                result = self.report(activities=activities)
                self.assertEqual(result["status"], "UNRESOLVED")
                self.assertIn("MALFORMED_FEE", result["reason_codes"])
                self.assertFalse(result["pnl_is_final"])

    def test_aggregate_fee_unrelated_day_and_external_cash_are_unresolved(self):
        activities = deepcopy(self.activities)
        activities[-1]["date"] = "2026-10-15"
        result = self.report(activities=activities, now=self.now+timedelta(days=1))
        self.assertIn("UNALLOCATED_FEE", result["reason_codes"])
        activities = self.activities+[dict(id="deposit", activity_type="CSD", date="2026-10-14", net_amount="100")]
        result = self.report(activities=activities)
        self.assertIn("EXTERNAL_ACTIVITY", result["reason_codes"])

    def test_account_currency_identity_cash_position_and_open_orders_mismatches(self):
        for kwargs, reason in ((dict(account=dict(self.account, account_id="other")), "MALFORMED_ACCOUNT"),
                               (dict(account=dict(self.account, currency="EUR")), "MALFORMED_ACCOUNT"),
                               (dict(account=dict(self.account, cash="100000")), "CASH_MISMATCH"),
                               (dict(positions=[dict(symbol="BTCUSD", qty=".1")]), "POSITION_MISMATCH"),
                               (dict(positions=[dict(symbol="ETHUSD", qty="1")]), "EXTERNAL_POSITION"),
                               (dict(open_orders=[dict(id="open")]), "OPEN_ORDERS")):
            with self.subTest(reason=reason):
                result = self.report(**kwargs)
                self.assertEqual(result["status"], "UNRESOLVED")
                self.assertIn(reason, result["reason_codes"])
                self.assertFalse(result["balances_match"])

    def test_extra_fill_or_changed_value_is_unresolved(self):
        for qty, price in (("1.1", "100"), ("1", "101")):
            activities = deepcopy(self.activities)
            activities[0].update(qty=qty, price=price)
            result = self.report(activities=activities)
            self.assertIn("FILL_MISMATCH", result["reason_codes"])
            self.assertFalse(result["fills_match"])

    def test_partial_fill_deltas_and_activity_aggregation_match(self):
        events = [self.event("buy", "BUY", ".4", "100", 0),
                  self.event("buy", "BUY", ".6", "100", 10), self.events[1]]
        activities = [self.fill("buy-first", "buy", "buy", ".4", "100", 0),
                      self.fill("buy-second", "buy", "buy", ".6", "100", 10)]+self.activities[1:]
        self.assertEqual(self.report(events=events, activities=activities)["status"], "RECONCILED_AS_OF")

    def test_no_trades_is_reconciled_as_of_if_account_clean(self):
        self.state.cash = "1000"
        result = self.report(events=[], activities=[], account=dict(self.account, cash="100000"))
        self.assertEqual(result["status"], "RECONCILED_AS_OF")
        self.assertEqual(result["net_pnl"], "0")
        self.assertEqual(result["reason_codes"], ["NO_TRADES"])
        self.assertFalse(result["has_posted_fees"])

    def test_report_hash_and_partial_fee_total_are_revisionable(self):
        first = self.report()
        activities = deepcopy(self.activities)
        activities[-1]["net_amount"] = "-.6"
        second = self.report(activities=activities, account=dict(self.account, cash="100008.3"))
        self.assertEqual(second["status"], "RECONCILED_AS_OF")
        self.assertNotEqual(first["activities_sha256"], second["activities_sha256"])
        self.assertEqual(D(second["net_pnl"]), D("8.3"))

    def test_pending_order_or_missing_journal_cannot_finalize(self):
        self.state.broker["pending"] = {"phase": "UNKNOWN"}
        result = self.report()
        self.assertIn("PENDING_ORDER", result["reason_codes"])
        self.assertFalse(result["pnl_is_final"])
        self.state.broker["pending"] = None
        events = [dict(kind="ORDER_UPDATE", occurred_at=stamp(self.day),
                       payload=dict(order_id="buy", side="buy", filled_qty="1"))]
        result = self.report(events=events)
        self.assertIn("FILL_WITHOUT_JOURNAL", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
