from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from traid_ai.broker import BrokerEngine
from traid_ai.dashboard import parse_settings, SessionManager
from traid_ai.engine import Engine
from traid_ai.models import Config,D,Quote,State,encode,parse_time,stamp,utc_now
from traid_ai.objectives import session_objective
from traid_ai.qwen import build_snapshot
from traid_ai.risk import entry_quantity
from traid_ai.store import Store
from traid_ai.strategy import Intent
from test_broker import FakeBroker,EntryRule


class TrailingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"state.sqlite3"
        self.store=Store(self.path);self.now=utc_now()
        self.config=Config(source="synthetic",target_return=D(".01"),max_hold_seconds=3600)
        self.engine=Engine.start(self.store,self.config,self.now,self.now+timedelta(hours=4))
        self.engine.strategy=EntryRule()

    def tearDown(self):
        self.store.close();self.tmp.cleanup()

    def tick(self, price="100", *, ask=None, age=0, **kwargs):
        self.now+=timedelta(seconds=1)
        return self.engine.step(self.now,Quote(stamp(self.now),stamp(self.now-timedelta(seconds=age)),
             D(price),D(ask or price),"synthetic"),**kwargs)

    def test_stop_follows_bid_high_water_and_never_falls(self):
        self.tick();self.tick("110")
        self.assertEqual(D(self.engine.state.position_peak),110)
        self.assertEqual(D(self.engine.state.stop_price),D("107.8"))
        self.tick("109",ask="999")
        self.assertEqual(D(self.engine.state.position_peak),110)
        self.assertEqual(D(self.engine.state.stop_price),D("107.8"))
        self.assertEqual(self.engine.state.exits,0)

    def test_stop_triggers_without_strategy_and_during_pause(self):
        self.tick();self.tick("110")
        self.engine.strategy.decide=lambda *a:(_ for _ in ()).throw(AssertionError("Must not call strategy"))
        events=self.tick("107",pause_entries=True)
        fills=[p for k,p in events if k=="PAPER_FILL"]
        self.assertEqual(fills[0]["reason"],"TRAILING_STOP")
        self.assertEqual(self.engine.state.status,"PAUSED_ENTRIES")
        self.assertEqual(D(self.engine.state.quantity),0)
        self.assertEqual(D(self.engine.state.position_peak),0)

    def test_strategy_cannot_preempt_primary_trailing_exit(self):
        self.tick()
        calls=[]
        self.engine.strategy.decide=lambda *a:calls.append(True) or Intent("REQUEST_EXIT","MODEL_EXIT")
        self.tick("101")
        self.assertEqual(calls,[])
        self.assertEqual(self.engine.state.exits,0)

    def test_rejected_quote_cannot_move_or_trigger_stop(self):
        self.tick();self.tick("110")
        before=(self.engine.state.position_peak,self.engine.state.stop_price)
        events=self.tick("200",age=60)
        self.assertTrue(any(k=="DATA_REJECTED" for k,p in events))
        self.assertEqual(before,(self.engine.state.position_peak,self.engine.state.stop_price))
        self.tick("90",age=60)
        self.assertEqual(self.engine.state.exits,0)

    def test_restart_restores_stop_and_peak(self):
        self.tick();self.tick("110")
        sid=self.engine.state.session_id
        self.store.close();self.store=Store(self.path);self.engine=Engine(self.store,sid)
        self.tick("108")
        self.assertEqual(D(self.engine.state.stop_price),D("107.8"))
        events=self.tick("107")
        self.assertTrue(any(k=="PAPER_FILL" and p["reason"]=="TRAILING_STOP" for k,p in events))

    def test_price_gap_does_not_invent_fill_at_stop_price(self):
        self.tick();self.tick("110")
        stop=D(self.engine.state.stop_price)
        events=self.tick("90")
        fill=next(p for k,p in events if k=="PAPER_FILL")
        self.assertLess(D(fill["price"]),stop)
        self.assertEqual(D(fill["price"]),D("90")*(1-self.config.slippage_bps/10000))

    def test_next_position_resets_high_water(self):
        self.tick();self.tick("110");self.tick("107")
        self.tick("100")
        self.assertEqual(self.engine.state.entries,2)
        self.assertLess(D(self.engine.state.position_peak),110)
        self.assertLess(D(self.engine.state.stop_price),100)

    def test_maximum_holding_and_deadline_still_override_goal(self):
        self.tick()
        self.now+=timedelta(seconds=self.config.max_hold_seconds)
        events=self.tick("101")
        self.assertTrue(any(k=="PAPER_FILL" and p["reason"]=="MAX_HOLD" for k,p in events))
        self.tick()
        self.now=parse_time(self.engine.state.deadline_at)-timedelta(seconds=60)
        self.tick("100")
        self.assertEqual(self.engine.state.status,"CLOSED")
        self.assertEqual(self.engine.state.exit_reason,"DEADLINE")

    def test_goal_uses_estimated_exit_costs_and_matches_actual_local_close(self):
        self.tick();self.tick("105")
        expected=D(self.engine.report()["objective"]["estimated_net_pnl_usd"])
        self.tick("105",stop=True)
        self.assertEqual(D(self.engine.state.realized_pnl),expected)
        self.assertEqual(D(self.engine.report()["objective"]["estimated_net_pnl_usd"]),expected)

    def test_goal_reached_does_not_force_exit_or_change_limits(self):
        self.tick();self.tick("110")
        report=self.engine.report()
        self.assertTrue(report["objective"]["reached"])
        self.assertEqual(self.engine.state.status,"RUNNING")
        self.assertEqual(self.engine.state.exits,0)
        self.assertEqual(self.engine.config.risk_per_trade,self.config.risk_per_trade)
        self.assertEqual(self.engine.config.max_exposure,self.config.max_exposure)

    def test_goal_has_no_effect_on_risk_quantity_and_missing_quote_is_unknown(self):
        args=[D("1000"),D("1000"),D("100"),D("98")]
        self.assertEqual(entry_quantity(self.config,*args),
                         entry_quantity(replace(self.config,target_return=D("10")),*args))
        state=deepcopy(self.engine.state);state.quantity="1";state.last_quote=None
        goal=session_objective(self.config,state)
        self.assertIsNone(goal["estimated_net_pnl_usd"])
        self.assertIsNone(goal["reached"])
        self.assertFalse(goal["mandatory"])
        self.assertFalse(goal["may_raise_risk"])
        self.assertFalse(goal["may_extend_deadline"])

    def test_zero_goal_and_no_goal_are_distinct(self):
        goal=session_objective(replace(self.config,target_return=D("0")),self.engine.state)
        self.assertTrue(goal["reached"])
        self.assertIsNone(goal["progress_percent"])
        self.assertIsNone(session_objective(replace(self.config,target_return=None),self.engine.state))

    def test_model_receives_goal_and_cannot_choose_discretionary_exit(self):
        config=replace(self.config,source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
        state=deepcopy(self.engine.state);state.quantity="1";state.entry_price="100";state.position_peak="110"
        q=Quote("model",stamp(self.now),D("108"),D("108.1"),"kraken-public")
        state.last_quote=dict(quote_id=q.quote_id,observed_at=q.observed_at,bid=str(q.bid),ask=str(q.ask),source=q.source)
        snapshot=build_snapshot(config,state,q,self.now)
        self.assertEqual(snapshot["snapshot_version"],"market-summary-v7")
        self.assertEqual(snapshot["objective"]["target_percent"],"1.00")
        self.assertEqual(snapshot["position_management"]["owner"],"INDEPENDENT_CODE")
        self.assertNotIn("REQUEST_EXIT",snapshot["allowed_actions"])
        other=build_snapshot(replace(config,target_return=D("1")),state,q,self.now)
        self.assertEqual(snapshot["decision_contract"],other["decision_contract"])

    def test_old_json_keeps_fixed_stop_and_has_no_target(self):
        payload=self.config.to_dict();payload.pop("exit_policy");payload.pop("target_return")
        old=Config.from_dict(payload)
        self.assertEqual(old.exit_policy,"fixed-stop-v1")
        self.assertIsNone(old.target_return)
        state=self.engine.state.to_dict();state.pop("position_peak");state.pop("position_exit_reason")
        self.assertEqual(State(**state).position_peak,"0")

    def test_qwen_v7_target_flows_through_risk_and_replay(self):
        from test_wait_review import WaitReviewTests
        from traid_ai.evaluation import compare_session
        fixture=WaitReviewTests();fixture.setUp()
        try:
            cfg=replace(fixture.config,exit_policy="trailing-stop-v1",target_return=D(".015"))
            with fixture.store.db:
                fixture.store.db.execute("UPDATE sessions SET config=? WHERE id=?",
                    (encode(cfg.to_dict()),fixture.engine.state.session_id))
            fixture.client.action="REQUEST_ENTRY"
            fixture.tick();fixture.done();fixture.tick()
            snapshot=fixture.client.calls[0]
            self.assertEqual(snapshot["objective"]["target_pnl_usd"],"15.000")
            self.assertEqual(snapshot["snapshot_version"],"market-summary-v7")
            self.assertEqual(fixture.engine.state.ai["prompt_version"],"qwen-trader-v7")
            self.assertEqual(fixture.engine.state.entries,1)
            self.assertEqual(fixture.engine.config.risk_per_trade,cfg.risk_per_trade)
            export=dict(config=cfg.to_dict(),state=fixture.engine.state.to_dict(),
                        events=fixture.store.events(fixture.engine.state.session_id))
            self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],1)
            export["config"]["target_return"]=D(".9")
            self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],0)
        finally:fixture.tearDown()

    def test_settings_validate_and_persist_target_and_stop(self):
        cfg,_=parse_settings(dict(mode="paper",target_percent="1.5",stop_percent="0.8"))
        self.assertEqual(cfg.target_return,D(".015"));self.assertEqual(cfg.stop_fraction,D(".008"))
        self.assertIsNone(parse_settings(dict(mode="paper",target_percent=""))[0].target_return)
        for patch in (dict(target_percent="NaN"),dict(target_percent="-1"),dict(target_percent="Infinity"),
                      dict(stop_percent="0"),dict(stop_percent="100")):
            with self.subTest(patch=patch),self.assertRaises(ValueError):parse_settings(dict(mode="paper",**patch))
        manager=SessionManager(Path(self.tmp.name)/"ui.sqlite3",demo_interval=.2)
        try:
            sid=manager.launch(payload=dict(mode="demo",duration_minutes="5",target_percent="1.5",stop_percent="0.8"))
            view=manager.view(sid)
            self.assertEqual(view["config"]["target_return"],D(".015"))
            self.assertEqual(view["report"]["objective"]["target_pnl_usd"],"15.000")
        finally:manager.shutdown()

    def test_target_above_100_percent_is_accepted_and_restored(self):
        for percent in ("101", "250", "1000"):
            with self.subTest(percent=percent):
                cfg,_=parse_settings(dict(mode="paper",target_percent=percent))
                restored=Config.from_dict(cfg.to_dict())
                self.assertEqual(restored.target_return,D(percent)/100)
                goal=session_objective(restored,self.engine.state)
                self.assertEqual(D(goal["target_pnl_usd"]),cfg.capital*D(percent)/100)
                self.assertFalse(goal["may_raise_risk"])


class BrokerTrailingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(Path(self.tmp.name)/"paper.sqlite3")
        self.client=FakeBroker();self.now=utc_now()
        self.engine=BrokerEngine.start_paper(self.store,Config(source="alpaca-paper"),self.now,
                                            self.now+timedelta(hours=1),self.client)
        self.engine.strategy=EntryRule()

    def tearDown(self):
        self.engine.close();self.store.close();self.tmp.cleanup()

    def tick(self,price="100",**kwargs):
        self.now+=timedelta(seconds=1);self.client.ask=D(price)
        return self.engine.step(self.now,Quote(stamp(self.now),stamp(self.now),D(price),D(price),"alpaca-paper"),**kwargs)

    def partial(self):
        self.tick();pending=self.engine.state.broker["pending"]
        cid=pending["request"]["client_order_id"];qty=D(pending["request"]["qty"])
        self.client.fill(cid,qty/2);self.tick("110")
        return cid,qty

    def test_later_partial_fill_cannot_lower_raised_stop(self):
        cid,qty=self.partial();before=D(self.engine.state.stop_price)
        self.client.fill(cid,qty/2);self.tick("109")
        self.assertEqual(D(self.engine.state.stop_price),before)
        self.assertEqual(D(self.engine.state.position_peak),110)

    def test_cancel_is_not_repeated_before_another_reconciliation(self):
        self.partial()
        self.engine._cancel(self.now)
        self.engine._cancel(self.now)
        self.assertEqual(len(self.client.cancels),1)

    def test_stop_cancels_remainder_and_latches_exit_after_price_rebound(self):
        cid,qty=self.partial();self.tick("107")
        self.assertTrue(self.client.cancels)
        self.assertEqual(self.engine.state.position_exit_reason,"TRAILING_STOP")
        self.assertEqual(len(self.client.posts),1)
        self.tick("109")
        self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])
        self.assertEqual(D(self.client.posts[-1]["qty"]),self.client.qty)

    def test_cancel_fill_race_and_restart_keep_single_reduction_order(self):
        cid,qty=self.partial();self.client.cancel_race=True;self.tick("107")
        before=self.engine.state.stop_price;sid=self.engine.state.session_id
        self.engine.close();self.engine=BrokerEngine(self.store,sid,self.client)
        self.tick("109")
        self.assertEqual([p["side"] for p in self.client.posts],["buy","sell"])
        self.assertGreaterEqual(D(self.engine.state.stop_price),D(before))
        sell=self.client.posts[-1]
        self.client.fill(sell["client_order_id"],D(sell["qty"]))
        self.tick("109")
        self.assertEqual(D(self.engine.state.quantity),0)
        self.assertEqual(self.engine.state.position_peak,"0")
        self.assertIsNone(self.engine.state.position_exit_reason)
        self.assertEqual(len(self.client.posts),2)
