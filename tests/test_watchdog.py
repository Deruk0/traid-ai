from datetime import datetime, timedelta, timezone
import threading
import unittest

from traid_ai.watchdog import Watchdog


class FakeClocks:
    def __init__(self):
        self.mono = 0
        self.wall = datetime(2026, 10, 14, tzinfo=timezone.utc)

    def advance(self, seconds, wall_seconds=None):
        self.mono += seconds
        self.wall += timedelta(seconds=seconds if wall_seconds is None else wall_seconds)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.clocks = FakeClocks()
        self.reasons = []
        self.watch = Watchdog(self.reasons.append, clock=lambda: self.clocks.mono,
                              wall_clock=lambda: self.clocks.wall)
        self.addCleanup(self.watch.stop)

    def test_idle_start_and_ten_second_worker_wait_are_healthy(self):
        self.assertEqual(self.watch.check()["status"], "IDLE")
        self.watch.start()
        thread = self.watch._thread
        self.watch.start()
        self.assertIs(self.watch._thread, thread)
        for _ in range(6):
            self.clocks.advance(10)
            self.assertEqual(self.watch.check()["status"], "HEALTHY")
            self.watch.progress()
        self.assertEqual(self.reasons, [])
        self.assertEqual(self.watch.view()["age_seconds"], 0)

    def test_network_stall_detected_without_monitor_stall(self):
        self.watch.start()
        self.watch.progress()
        for _ in range(4):
            self.clocks.advance(10)
            self.assertEqual(self.watch.check()["status"], "HEALTHY")
        self.clocks.advance(10)
        view = self.watch.check()
        self.assertEqual(view["status"], "STALLED")
        self.assertEqual(view["reason"], "WORKER_STALLED")
        self.assertEqual(view["age_seconds"], 50)
        self.assertIsNotNone(view["detected_at"])
        self.assertEqual(self.reasons, ["WORKER_STALLED"])

    def test_monitor_gap_twenty_to_sixty_and_resume_remain_latched(self):
        self.watch.start()
        self.clocks.advance(20)
        self.watch.check()
        self.watch.progress()
        self.clocks.advance(40)
        self.assertEqual(self.watch.check()["status"], "RESUME_GAP")
        self.watch.progress()
        for _ in range(3):
            self.watch.check()
        self.assertTrue(self.watch.view()["tripped"])
        self.assertEqual(self.reasons, ["SYSTEM_RESUME_GAP"])

    def test_wall_clock_detects_sleep_when_monotonic_does_not_advance(self):
        self.watch.start()
        self.clocks.advance(1, wall_seconds=60)
        self.assertEqual(self.watch.check()["reason"], "SYSTEM_RESUME_GAP")

    def test_clock_discrepancy_detected_for_backward_wall_adjustment(self):
        self.watch.start()
        self.clocks.advance(1, wall_seconds=-60)
        self.assertEqual(self.watch.check()["reason"], "SYSTEM_RESUME_GAP")

    def test_callback_runs_outside_lock_and_exception_is_redacted(self):
        def trip(reason):
            self.assertTrue(self.watch.view()["tripped"])
            self.assertEqual(reason, "SYSTEM_RESUME_GAP")
            raise RuntimeError("secret-value")
        self.watch._on_trip = trip
        self.watch.start()
        self.clocks.advance(60)
        view = self.watch.check()
        self.assertEqual(view["callback_error"], "CALLBACK_FAILED")
        self.assertNotIn("secret-value", str(view))

    def test_stop_joins_monitor_and_does_not_clear_trip(self):
        self.watch.start()
        self.clocks.advance(60)
        self.watch.check()
        thread = self.watch._thread
        self.watch.stop()
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.watch.view()["status"], "STOPPED")
        self.assertTrue(self.watch.check()["tripped"])
        self.watch.progress()
        self.assertEqual(self.watch.view()["status"], "STOPPED")

    def test_actual_monitor_thread_signals_only_callback(self):
        signal = threading.Event()
        watch = Watchdog(lambda reason: signal.set(), stale_seconds=0.04,
                         resume_gap_seconds=0.5)
        self.addCleanup(watch.stop)
        watch.start()
        self.assertTrue(signal.wait(1))
        self.assertEqual(watch.view()["reason"], "WORKER_STALLED")
        watch.stop()
        self.assertFalse(watch._thread.is_alive())

    def test_invalid_thresholds_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Watchdog(self.reasons.append, stale_seconds=value)
            with self.assertRaises(ValueError):
                Watchdog(self.reasons.append, resume_gap_seconds=value)


if __name__ == "__main__":
    unittest.main()
