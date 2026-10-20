from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time
import unittest

from traid_ai.experiment import Experiment
from traid_ai.models import Config, D, Quote, stamp, encode
from traid_ai.store import Store


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DIGEST = "a" * 64


class ControlledModel:
    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.calls = []

    def decide(self, snapshot, *, digest, timeout=20):
        self.calls.append((snapshot, digest))
        self.release.wait(5)
        return dict(snapshot_id=snapshot["snapshot_id"], action="REQUEST_ENTRY", reason="Контроль",
                    latency_ms=10)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "experiments.sqlite3")
        self.model = ControlledModel()
        self.config = Config(source="kraken-public", strategy_version="qwen-local-v1", llm_digest=DIGEST,decision_policy="legacy")
        self.experiment = Experiment.start(self.store, self.config, NOW, NOW + timedelta(hours=1), client=self.model)

    def tearDown(self):
        self.model.release.set()
        self.experiment.close()
        self.store.close()
        self.tmp.cleanup()

    def tick(self, seconds, price="100", **kwargs):
        now = NOW + timedelta(seconds=seconds)
        quote = Quote(str(seconds), stamp(now), D(price), D(price) + D(".01"), "kraken-public")
        return self.experiment.step(now, quote, **kwargs)

    def wait_model(self):
        end = time.monotonic() + 2
        strategy = self.experiment.engines["qwen"].strategy
        while time.monotonic() < end:
            if strategy.job and strategy.job["done"]:
                return
            time.sleep(.005)
        self.fail("Фоновый ответ не завершился")

    def test_real_background_strategy_same_quotes_independent_portfolios(self):
        for seconds in range(1, 7):
            self.tick(seconds)
        self.wait_model()
        self.tick(7)
        report = self.experiment.report()
        self.assertEqual(report["common_quote_count"], 7)
        self.assertEqual(report["all_arms_accepted_quote_count"], 7)
        self.assertEqual(report["results"]["qwen"]["entries"], 1)
        self.assertEqual(report["results"]["demo_momentum"]["entries"], 0)
        self.assertEqual(report["results"]["usd"]["entries"], 0)
        self.assertEqual(report["results"]["qwen"]["mean_response_latency_ms"], "10")
        self.assertEqual(report["evidence"], "UNASSESSED")
        snapshot, digest = self.model.calls[0]
        self.assertEqual(digest, DIGEST)
        self.assertNotIn("broker", encode(snapshot))
        self.assertNotIn("api_key", encode(snapshot))
        self.assertTrue(all(engine.config.capital == self.config.capital for engine in self.experiment.engines.values()))

    def test_slow_model_does_not_block_stop_or_deadline(self):
        for stop in (True, False):
            model = ControlledModel()
            model.release.clear()
            experiment = Experiment.start(self.store, self.config, NOW, NOW + timedelta(hours=1), client=model)
            try:
                for seconds in range(1, 7):
                    when = NOW + timedelta(seconds=seconds)
                    experiment.step(when, Quote(str(seconds), stamp(when), D("100"), D("100.01"), "kraken-public"))
                before = time.monotonic()
                when = NOW + timedelta(seconds=8 if stop else 3600)
                experiment.step(when, Quote("stop", stamp(when), D("100"), D("100.01"), "kraken-public"), stop=stop)
                self.assertLess(time.monotonic() - before, 1)
                self.assertTrue(all(e.state.status == "CLOSED" for e in experiment.engines.values()))
                model.release.set()
                experiment.step(when + timedelta(seconds=1), None)
                self.assertEqual(experiment.report()["results"]["qwen"]["entries"], 0)
            finally:
                model.release.set()
                experiment.close()

    def test_hash_independent_of_ids_and_report_persistence(self):
        other = Experiment.start(self.store, self.config, NOW, NOW + timedelta(hours=1), client=ControlledModel())
        try:
            for seconds in range(1, 4):
                self.tick(seconds)
                when = NOW + timedelta(seconds=seconds)
                other.step(when, Quote("other-" + str(seconds), stamp(when), D("100"), D("100.01"), "kraken-public"))
            self.assertEqual(other.report()["dataset_sha256"], self.experiment.report()["dataset_sha256"])
            report = self.experiment.persist_report()
            for role, engine in self.experiment.engines.items():
                persisted = self.store.analysis(engine.state.session_id, "experiment")
                self.assertEqual(persisted["dataset_sha256"], report["dataset_sha256"])
                marker = self.store.analysis(engine.state.session_id, "experiment_role")
                self.assertEqual(marker["role"], role)
                self.assertFalse(marker["resume_allowed"])
        finally:
            other.close()

    def test_duplicate_step_idempotent_and_invalid_data_shared(self):
        self.tick(1)
        self.assertEqual(self.tick(1), {})
        when = NOW + timedelta(seconds=2)
        self.experiment.step(when, Quote("stale", stamp(NOW - timedelta(hours=1)), D("100"), D("100.01"), "kraken-public"))
        report = self.experiment.report()
        self.assertEqual(report["common_quote_count"], 2)
        self.assertTrue(all(row["quote_count"] == 1 and row["data_rejected_count"] == 1 for row in report["results"].values()))

    def test_reject_broker_synthetic_missing_digest_and_bad_duration(self):
        for source in ("alpaca-paper", "synthetic"):
            config = Config(source=source, llm_digest=DIGEST)
            with self.assertRaises(ValueError):
                Experiment.start(self.store, config, NOW, NOW + timedelta(hours=1))
        with self.assertRaises(ValueError):
            Experiment.start(self.store, Config(source="kraken-public"), NOW, NOW + timedelta(hours=1))
        with self.assertRaises(ValueError):
            Experiment.start(self.store, self.config, NOW, NOW + timedelta(seconds=60))

    def test_common_cost_policy_and_empty_coverage_not_fake_zero(self):
        for seconds in range(1, 6):
            self.tick(seconds)
        self.tick(6, price="101")
        self.wait_model()
        self.tick(7, price="101")
        report = self.experiment.report()
        self.assertEqual(report["results"]["demo_momentum"]["fees"], report["results"]["qwen"]["fees"])
        self.assertEqual(report["results"]["demo_momentum"]["quantity"], report["results"]["qwen"]["quantity"])
        self.assertIsNone(report["results"]["usd"]["ai_response_coverage"])
        self.assertIsNone(report["results"]["usd"]["ai_rejected_percent"])

    def test_close_disallows_steps_and_time_cannot_go_back(self):
        self.tick(2)
        with self.assertRaises(ValueError):
            self.tick(1)
        self.experiment.close()
        with self.assertRaises(ValueError):
            self.tick(3)


if __name__ == "__main__":
    unittest.main()
