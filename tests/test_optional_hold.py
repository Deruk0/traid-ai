"""New sessions can omit MAX_HOLD while saved sessions retain their limits."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from traid_ai.__main__ import parser,settings
from traid_ai.dashboard import parse_settings
from traid_ai.engine import Engine
from traid_ai.models import Config,D,Quote,stamp,utc_now
from traid_ai.qwen import build_snapshot
from traid_ai.evaluation import compare_session
from traid_ai.store import Store
from test_broker import EntryRule
from test_evaluation import dataset,add_decision


class OptionalHoldTests(unittest.TestCase):
    def test_new_config_defaults_to_disabled_and_old_missing_field_retains_one_hour(self):
        config=Config(source="synthetic")
        self.assertIsNone(config.max_hold_seconds)
        self.assertIsNone(Config.from_dict(config.to_dict()).max_hold_seconds)
        old=config.to_dict();old.pop("max_hold_seconds")
        self.assertEqual(Config.from_dict(old).max_hold_seconds,3600)
        self.assertEqual(Config.from_dict(dict(old,max_hold_seconds=7200)).max_hold_seconds,7200)
        for bad in (0,-1,True,1.5,float("nan"),"3600"):
            with self.subTest(value=bad),self.assertRaises(ValueError):replace(config,max_hold_seconds=bad)

    def test_ui_and_cli_accept_empty_or_explicit_limit(self):
        for value in (None,""):
            self.assertIsNone(parse_settings(dict(mode="paper",max_hold_minutes=value))[0].max_hold_seconds)
        self.assertEqual(parse_settings(dict(mode="paper",max_hold_minutes="90"))[0].max_hold_seconds,5400)
        self.assertIsNone(settings(parser().parse_args(["paper"]),"kraken-public").max_hold_seconds)
        self.assertEqual(settings(parser().parse_args(["paper","--max-hold-minutes","90"]),"kraken-public").max_hold_seconds,5400)
        for value in ("0","-1","nan","0.001"):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_settings(dict(mode="paper",max_hold_minutes=value))

    def test_disabled_hold_keeps_position_after_hour_but_deadline_still_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/"hold.sqlite3");now=utc_now()
            engine=Engine.start(store,Config(source="synthetic"),now,now+timedelta(hours=3))
            engine.strategy=EntryRule()
            def tick(when):return engine.step(when,Quote(stamp(when),stamp(when),D("100"),D("100"),"synthetic"))
            tick(now)
            events=tick(now+timedelta(hours=2))
            self.assertGreater(D(engine.state.quantity),0)
            self.assertFalse(any(k=="PAPER_FILL" and p.get("side")=="SELL" for k,p in events))
            events=tick(now+timedelta(hours=3))
            self.assertEqual(engine.state.status,"CLOSED")
            self.assertTrue(any(k=="PAPER_FILL" and p.get("reason")=="DEADLINE" for k,p in events))
            store.close()

    def test_modern_snapshot_binds_hold_limit_and_replay_rejects_change(self):
        now=utc_now();config=Config(source="kraken-public",strategy_version="qwen-local-v1",llm_digest="a"*64)
        from traid_ai.models import State
        state=State("hold",stamp(now),stamp(now+timedelta(hours=2)),"RUNNING","1000","1000")
        quote=Quote("q",stamp(now),D("100"),D("100.01"),config.source)
        off=build_snapshot(config,state,quote,now)
        finite=build_snapshot(replace(config,max_hold_seconds=3600),state,quote,now)
        self.assertNotEqual(off["snapshot_id"],finite["snapshot_id"])
        self.assertIsNone(off["policy"]["max_hold_seconds"])
        # Existing legacy snapshots and old configs are still accepted unchanged.
        export=dataset(qwen=True);export["config"]["max_hold_seconds"]=3600
        add_decision(export)
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"],1)
        modern=dataset(qwen=True);modern["config"]["decision_policy"]="review-v1"
        request,answer=add_decision(modern,action="WAIT")
        snapshot=request["payload"]["snapshot"]
        choice=next(o for o in snapshot["decision_contract"] if o["action"]=="WAIT")
        answer["payload"]["answer"]=dict(snapshot_id=snapshot["snapshot_id"],**choice)
        self.assertEqual(compare_session(modern)["qwen_replay"]["replayed_decisions"],1)
        modern["config"]["max_hold_seconds"]=3600
        self.assertEqual(compare_session(modern)["qwen_replay"]["replayed_decisions"],0)
