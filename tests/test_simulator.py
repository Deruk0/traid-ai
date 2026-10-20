from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from traid_ai.engine import Engine
from traid_ai.feeds import parse_kraken
from traid_ai.models import Config, D, Quote, stamp
from traid_ai.risk import entry_quantity
from traid_ai.store import Store, writer_lock


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "paper.sqlite3"
        self.store = Store(self.path)
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.config = Config(source="synthetic",exit_policy="fixed-stop-v1")
        self.engine = Engine.start(self.store, self.config, self.start, self.start + timedelta(hours=1))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def quote(self, index, price="100", age=0):
        now = self.start + timedelta(seconds=10 * index)
        return now, Quote(f"q{index}", stamp(now - timedelta(seconds=age)), D(price),
                          D(price) + D("0.1"), "synthetic")

    def enter(self):
        for index in range(5):
            self.engine.step(*self.quote(index))
        self.engine.step(*self.quote(5, "102"))
        self.assertGreater(D(self.engine.state.quantity), 0)

    def test_risk_quantity_fits_cash_and_planned_loss(self):
        qty = entry_quantity(self.config, D("1000"), D("1000"), D("102"), D("99.96"))
        self.assertGreater(qty, 0)
        self.assertLessEqual(qty * D("102") * D("1.001"), D("250"))
        exit_price = D("99.96") * D("0.9995")
        loss = qty * (D("102") - exit_price + D("0.001") * (D("102") + exit_price))
        self.assertLessEqual(loss, D("5"))
        self.assertEqual(qty % self.config.quantity_step, 0)

    def test_duplicate_cannot_double_fill_or_charge(self):
        self.enter()
        before = self.engine.state.to_dict()
        event_count = len(self.store.events(self.engine.state.session_id))
        self.assertEqual(self.engine.step(*self.quote(5, "102")), [])
        self.assertEqual(self.engine.state.to_dict(), before)
        self.assertEqual(len(self.store.events(self.engine.state.session_id)), event_count)

    def test_restart_and_deadline_exit_conserve_cash(self):
        self.enter()
        session_id = self.engine.state.session_id
        self.store.close()
        self.store = Store(self.path)
        recovered = Engine(self.store, session_id)
        now = self.start + timedelta(minutes=59)
        recovered.step(now, Quote("end", stamp(now), D("105"), D("105.1"), "synthetic"))
        self.assertEqual(recovered.state.status, "CLOSED")
        self.assertEqual(D(recovered.state.quantity), 0)
        self.assertEqual(D(recovered.state.cash) - self.config.capital, D(recovered.state.realized_pnl))
        self.assertEqual((recovered.state.entries, recovered.state.exits), (1, 1))
        self.assertGreater(D(recovered.state.fees), 0)

    def test_missing_quote_at_deadline_is_unresolved_then_recovers(self):
        self.enter()
        now = self.start + timedelta(hours=1)
        self.engine.step(now, None, error="offline", step_id="deadline")
        self.assertEqual(self.engine.state.status, "UNRESOLVED")
        self.assertGreater(D(self.engine.state.quantity), 0)
        self.assertEqual(self.engine.state.exits, 0)
        self.engine.step(now + timedelta(seconds=10), Quote("recovery", stamp(now + timedelta(seconds=10)),
                                                          D("101"), D("101.1"), "synthetic"))
        self.assertEqual(self.engine.state.status, "CLOSED")

    def test_stale_and_wrong_source_block_entry(self):
        for i in range(5):
            self.engine.step(*self.quote(i))
        self.engine.step(*self.quote(5, "102", age=31))
        now, quote = self.quote(6, "103")
        self.engine.step(now, replace(quote, source="kraken-public"))
        self.assertEqual(self.engine.state.entries, 0)
        self.assertEqual(sum(e["kind"] == "DATA_REJECTED" for e in self.store.events(self.engine.state.session_id)), 2)

    def test_loss_limit_closes_and_disables_new_entries(self):
        self.enter()
        self.engine.step(*self.quote(6, "50"))
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(self.engine.state.exit_reason, "SESSION_LOSS_LIMIT")
        self.engine.step(*self.quote(7, "200"))
        self.assertEqual(self.engine.state.entries, 1)

    def test_stop_loss_is_independent_of_strategy(self):
        self.enter()
        self.engine.strategy.decide = lambda *args: (_ for _ in ()).throw(AssertionError("strategy called"))
        # Protection should not depend on even calling the decision provider.
        self.engine.step(*self.quote(6, "98"))
        fills = [e for e in self.store.events(self.engine.state.session_id) if e["kind"] == "PAPER_FILL"]
        self.assertEqual(fills[-1]["payload"]["reason"], "STOP_LOSS")

    def test_no_entry_during_flatten_window(self):
        for i in range(5):
            self.engine.step(*self.quote(i))
        now = self.start + timedelta(minutes=59)
        self.engine.step(now, Quote("cutoff", stamp(now), D("110"), D("110.1"), "synthetic"))
        self.assertEqual(self.engine.state.status, "CLOSED")
        self.assertEqual(self.engine.state.entries, 0)

    def test_decision_failure_waits_but_deadline_still_exits(self):
        self.enter()
        self.engine.strategy.decide = lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))
        events = self.engine.step(*self.quote(6, "102"))
        self.assertTrue(any(kind == "DECISION_UNAVAILABLE" for kind, _ in events))
        self.assertEqual(self.engine.state.exits, 0)
        now = self.start + timedelta(minutes=59)
        self.engine.step(now, Quote("close", stamp(now), D("102"), D("102.1"), "synthetic"))
        self.assertEqual(self.engine.state.status, "CLOSED")

    def test_failed_transaction_reloads_state_before_retry(self):
        for index in range(5):
            self.engine.step(*self.quote(index))
        original = self.store.commit_step
        self.store.commit_step = lambda *args: (_ for _ in ()).throw(RuntimeError("disk"))
        with self.assertRaises(RuntimeError):
            self.engine.step(*self.quote(5, "102"))
        self.store.commit_step = original
        restored = Engine(self.store, self.engine.state.session_id)
        self.assertEqual(restored.state.entries, 0)
        self.engine.step(*self.quote(5, "102"))
        self.assertEqual(self.engine.state.entries, 1)

    def test_reordered_quotes_cannot_change_balance(self):
        self.enter()
        old_state = self.engine.state.to_dict()
        now, quote = self.quote(6, "500")
        quote = replace(quote, observed_at=stamp(self.start + timedelta(seconds=50)))
        self.engine.step(now, quote)
        self.assertEqual(self.engine.state.cash, old_state["cash"])
        self.assertEqual(self.engine.state.quantity, old_state["quantity"])

    def test_nonfinite_config_and_quotes_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.config, capital=D("NaN"))
        now, quote = self.quote(0)
        self.engine.step(now, replace(quote, ask=D("Infinity")))
        self.assertIsNone(self.engine.state.last_quote)

    def test_public_payload_rejects_crossed_book_and_wrong_pair(self):
        payload = {"error": [], "result": {"XXBTZUSD": {"b": ["100"], "a": ["101"]}}}
        quote = parse_kraken(payload, self.start)
        self.assertEqual(quote.bid, D("100"))
        payload["result"]["XXBTZUSD"]["a"] = ["99"]
        with self.assertRaises(ValueError):
            parse_kraken(payload, self.start)
        with self.assertRaises(ValueError):
            parse_kraken({"error": [], "result": {"ETHUSD": {}}}, self.start)

    def test_single_writer_lock_released_after_exit(self):
        script = "from pathlib import Path; from traid_ai.store import writer_lock; import sys\ntry:\n with writer_lock(Path(sys.argv[1])): pass\nexcept ValueError: sys.exit(7)"
        with writer_lock(self.path):
            result = subprocess.run([sys.executable, "-c", script, str(self.path)], capture_output=True)
            self.assertEqual(result.returncode, 7, result.stderr)
        with writer_lock(self.path):
            pass

    def test_cli_demo_and_report(self):
        db = Path(self.temp.name) / "cli.sqlite3"
        run = subprocess.run([sys.executable, "-m", "traid_ai", "demo", "--db", str(db)],
                             capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout.splitlines()[-1])
        self.assertEqual(report["status"], "CLOSED")
        self.assertGreater(report["entries"], 0)
        saved = subprocess.run([sys.executable, "-m", "traid_ai", "report", report["session_id"],
                                "--db", str(db)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(saved.returncode, 0, saved.stderr)
        self.assertEqual(json.loads(saved.stdout)["realized_pnl"], report["realized_pnl"])


if __name__ == "__main__":
    unittest.main()
