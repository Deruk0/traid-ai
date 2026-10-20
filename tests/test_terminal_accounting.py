"""Terminal pending orders must not exclude verified activity reconciliation."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import threading
import time
import unittest

from traid_ai.broker import BrokerEngine
from traid_ai.models import Config,D,Quote,stamp,utc_now
from traid_ai.store import Store
from test_alpaca import await_condition
from test_broker import FakeBroker,EntryRule
from unittest.mock import patch
from traid_ai.dashboard import SessionManager


class PostedFeeBroker(FakeBroker):
    def __init__(self,now):
        super().__init__()
        self.auto_fill=True
        self.now=now
        self.extra_fee=D("2")
        self.explain_extra=True
        self.buy_fee_qty=D(0)
        self.activity_reads=0
        self.release=threading.Event();self.release.set()

    def account(self):
        return dict(super().account(),currency="USD")

    def fill(self,cid,delta):
        before=self.qty
        super().fill(cid,delta)
        if self.records[cid]["side"]=="buy":
            self.buy_fee_qty+=delta-(self.qty-before)
        else:
            self.cash-=self.extra_fee

    def activities(self,after,until):
        self.activity_reads+=1
        self.release.wait(2)
        rows=[]
        for n,r in enumerate(self.records.values()):
            when=stamp(self.now+timedelta(seconds=n+1))
            rows.append(dict(id="fill-"+r["id"],activity_type="FILL",order_id=r["id"],
                             transaction_time=when,symbol="BTC/USD",side=r["side"],
                             qty=r["filled_qty"],price=r["filled_avg_price"]))
            if r["side"]=="buy":
                rows.append(dict(id="fee-buy",activity_type="CFEE",order_id=r["id"],
                                 transaction_time=when,symbol="BTC/USD",qty=str(-self.buy_fee_qty),
                                 price=r["filled_avg_price"],net_amount="0",status="executed"))
            else:
                fee=D(r["filled_qty"])*D(r["filled_avg_price"])*D(".0025")
                if self.explain_extra:fee+=self.extra_fee
                rows.append(dict(id="fee-sell",activity_type="FEE",order_id=r["id"],
                                 transaction_time=when,symbol="USD",currency="USD",qty="0",
                                 net_amount=str(-fee),status="executed"))
        return rows


class TerminalAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.now=utc_now()
        self.store=Store(Path(self.tmp.name)/"fees.sqlite3")
        self.client=PostedFeeBroker(self.now)
        self.engine=BrokerEngine.start_paper(self.store,Config(source="alpaca-paper"),self.now,
                                            self.now+timedelta(hours=2),self.client)
        self.engine.strategy=EntryRule()
        self.tick();self.tick();self.tick(stop=True)
        self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])

    def tearDown(self):
        self.client.release.set();self.engine.close();self.store.close();self.tmp.cleanup()

    def tick(self,**control):
        self.now+=timedelta(seconds=1)
        return self.engine.step(self.now,Quote(stamp(self.now),stamp(self.now),D("99.99"),D("100"),"alpaca-paper"),**control)

    def test_posted_fees_unlock_terminal_order_without_repeat_sell_or_double_charge(self):
        self.tick()
        self.assertIsNotNone(self.engine._activity_job)
        await_condition(lambda:self.engine._activity_job["done"])
        self.tick()
        self.assertEqual(self.engine.state.status,"CLOSED")
        self.assertEqual(D(self.engine.state.quantity),0)
        self.assertIsNone(self.engine.state.broker["pending"])
        self.assertTrue(self.engine.state.accounting["balances_match"])
        self.assertTrue(self.engine.state.accounting["fills_match"])
        cash=self.engine.config.capital+self.client.cash-D(self.engine.state.broker["initial_cash"])
        self.assertEqual(D(self.engine.state.cash),cash)
        self.assertEqual(D(self.engine.state.realized_pnl),cash-self.engine.config.capital)
        before=deepcopy(self.engine.state.to_dict())
        self.tick()
        self.assertEqual(self.engine.state.to_dict(),before)
        self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])

    def test_unexplained_cash_change_stays_unresolved_with_observed_mismatch(self):
        self.client.explain_extra=False
        self.tick();await_condition(lambda:self.engine._activity_job["done"])
        self.tick()
        self.assertEqual(self.engine.state.status,"UNRESOLVED")
        self.assertIsNotNone(self.engine.state.broker["pending"])
        diagnostic=self.engine.state.broker["reconciliation"]
        self.assertEqual(D(diagnostic["actual_quantity"]),0)
        self.assertTrue(diagnostic["quantity_matches"])
        self.assertFalse(diagnostic["cash_matches"])
        self.assertIn("CASH_MISMATCH",self.engine.state.accounting["reason_codes"])
        self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])

    def test_activity_read_does_not_block_or_release_unverified_position(self):
        self.client.release.clear()
        before=self.engine.state.quantity
        started=time.monotonic();self.tick()
        self.assertLess(time.monotonic()-started,.5)
        self.assertEqual(self.engine.state.status,"UNRESOLVED")
        self.assertEqual(self.engine.state.quantity,before)
        self.assertTrue(self.engine.state.broker["pending"]["terminal"])
        self.assertEqual(len(self.client.posts),2)


class UnresolvedMaintenanceTests(unittest.TestCase):
    def test_read_only_accounting_does_not_resume_or_rewrite_trading_state(self):
        from test_accounting import AccountingTests
        fixture=AccountingTests();fixture.setUp();fixture.state.status="UNRESOLVED"
        class ReadOnlyClient:
            def account(self):return deepcopy(fixture.account)
            def activities(self,after,until):return deepcopy(fixture.activities)
            def positions(self):return []
            def open_orders(self):return []
            def submit_order(self,*args):raise AssertionError("Read-only maintenance cannot trade")
            def cancel_order(self,*args):raise AssertionError("Read-only maintenance cannot cancel")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"readonly.sqlite3";store=Store(path)
            store.create(fixture.config,fixture.state)
            for i,event in enumerate(fixture.events):
                store.commit_step(fixture.state,str(i),event["occurred_at"],[(event["kind"],event["payload"])])
            before={k:fixture.state.to_dict()[k] for k in ("cash","quantity","status","deadline_at","broker")}
            store.close();manager=SessionManager(path)
            try:
                with patch.object(manager.alpaca,"execution_client",return_value=ReadOnlyClient()):
                    manager.analyze_session(fixture.state.session_id,"accounting")
                    await_condition(lambda:not manager.maintenance["running"])
                self.assertIsNone(manager.maintenance["error"])
                view=manager.view(fixture.state.session_id,full_events=True)
                self.assertEqual({k:view["state"][k] for k in before},before)
                self.assertTrue(view["state"]["accounting"]["balances_match"])
                self.assertEqual(D(view["state"]["accounting"]["actual_quantity"]),0)
                self.assertFalse(any(e["kind"]=="ORDER_SUBMITTING" for e in view["events"]))
                with self.assertRaises(ValueError):manager.analyze_session(fixture.state.session_id,"comparison")
            finally:manager.shutdown()
