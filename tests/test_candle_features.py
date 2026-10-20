from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from traid_ai.history import summarize_candles
from traid_ai.models import D, stamp


class CandleFeatureTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)

    def rows(self, closes, volumes=None):
        prices = [D(str(p)) for p in closes]
        count = len(prices)-1
        volumes = volumes if volumes is not None else [1]*count
        return [dict(source="kraken-public", instrument="BTC/USD",
                     start_at=stamp(self.now-timedelta(minutes=count-i)),
                     end_at=stamp(self.now-timedelta(minutes=count-i-1)), available_at=stamp(self.now),
                     open=str(a), high=str(max(a, b)), low=str(min(a, b)), close=str(b),
                     volume=str(volumes[i])) for i, (a, b) in enumerate(zip(prices, prices[1:]))]

    def summary(self, rows, now=None):
        return summarize_candles(rows, "kraken-public", now or self.now)

    def movement(self, closes):
        return self.summary(self.rows(closes))["windows"][1]["movement"]

    def volume_rows(self, current=7):
        return self.rows([100]*36, [i//5+1 for i in range(30)]+[current]*5)

    def test_equal_net_growth_distinguishes_smooth_jump_and_choppy_paths(self):
        smooth = self.movement([100, 102, 104, 106, 108, 110])
        jump = self.movement([100, 100, 100, 110, 110, 110])
        choppy = self.movement([100, 106, 101, 109, 102, 110])
        self.assertEqual(smooth["path_efficiency_pct"], "100.000000")
        self.assertEqual(smooth["largest_step_share_pct"], "20.000000")
        self.assertEqual(jump["path_efficiency_pct"], "100.000000")
        self.assertEqual(jump["largest_step_share_pct"], "100.000000")
        self.assertEqual(choppy["path_efficiency_pct"], "29.411765")
        self.assertEqual(choppy["largest_step_share_pct"], "23.529412")
        self.assertGreater(D(smooth["return_to_variation_ratio"]), D(choppy["return_to_variation_ratio"]))
        self.assertEqual(jump["return_to_variation_ratio"], "1.000000")

    def test_downtrend_preserves_direction_in_return_ratio(self):
        movement = self.movement([110, 108, 106, 104, 102, 100])
        self.assertEqual(movement["path_efficiency_pct"], "100.000000")
        self.assertEqual(movement["largest_step_share_pct"], "20.000000")
        self.assertEqual(movement["pullback_from_high_pct"], "9.090909")
        self.assertEqual(movement["close_position_in_range_pct"], "0.000000")
        self.assertLess(D(movement["return_to_variation_ratio"]), 0)

    def test_wick_high_and_low_define_pullback_and_range_position(self):
        rows = self.rows([100, 105, 104, 106, 103, 105])
        rows[1].update(high="110", low="90")
        movement = self.summary(rows)["windows"][1]["movement"]
        self.assertEqual(movement["pullback_from_high_pct"], "4.545455")
        self.assertEqual(movement["close_position_in_range_pct"], "75.000000")

    def test_flat_path_has_zero_pullback_and_undefined_normalized_ratios(self):
        rows = self.rows([100]*6)
        window = self.summary(rows)["windows"][1]
        self.assertEqual(window["status"], "READY")
        self.assertEqual(window["sampled_variation_pct"], "0.000000")
        self.assertEqual(window["movement"], dict(path_efficiency_pct=None, largest_step_share_pct=None,
            pullback_from_high_pct="0.000000", close_position_in_range_pct=None, return_to_variation_ratio=None))
        for row in rows:
            row.update(high="102", low="98")
        movement = self.summary(rows)["windows"][1]["movement"]
        self.assertIsNone(movement["path_efficiency_pct"])
        self.assertEqual(movement["close_position_in_range_pct"], "50.000000")

    def test_movement_is_absent_for_unavailable_partial_gapped_or_stale_windows(self):
        rows = self.rows([100, 101, 102, 103, 104, 105])
        gapped = deepcopy(rows)
        del gapped[2]
        cases = [(self.summary([]), "UNAVAILABLE"), (self.summary(rows[-4:]), "WARMUP"),
                 (self.summary(gapped), "GAPS"),
                 (self.summary(rows, self.now+timedelta(seconds=181)), "STALE")]
        for summary, status in cases:
            with self.subTest(status=status):
                self.assertEqual(summary["windows"][1]["status"], status)
                self.assertIsNone(summary["windows"][1]["movement"])

    def test_volume_compares_equal_disjoint_windows_excluding_current(self):
        rows = self.volume_rows()
        original = deepcopy(rows)
        summary = self.summary(rows)
        volume = summary["relative_volume"]
        self.assertEqual(summary["version"], "candle-analytics-v2")
        self.assertEqual(volume["status"], "READY")
        self.assertEqual(volume["current_volume"], "35.000000")
        self.assertEqual(volume["baseline_mean_volume"], "17.500000")
        self.assertEqual(volume["ratio"], "2.000000")
        self.assertEqual(volume["required_candles"], 35)
        self.assertEqual(volume["baseline_windows"], 6)
        self.assertEqual(volume["start_at"], rows[0]["start_at"])
        self.assertEqual(rows, original)
        doubled = self.summary(self.volume_rows(current=14))["relative_volume"]
        self.assertEqual(doubled["baseline_mean_volume"], volume["baseline_mean_volume"])
        self.assertEqual(doubled["ratio"], "4.000000")

    def test_volume_requires_all_six_prior_windows(self):
        rows = self.volume_rows()
        volume = self.summary(rows[1:])["relative_volume"]
        self.assertEqual(volume["status"], "WARMUP")
        self.assertEqual(volume["observed_candles"], 34)
        self.assertEqual(volume["current_volume"], "35.000000")
        self.assertIsNone(volume["baseline_mean_volume"])
        self.assertIsNone(volume["ratio"])
        self.assertEqual(self.summary([])["relative_volume"]["status"], "UNAVAILABLE")

    def test_gap_in_baseline_blocks_ratio_even_when_current_window_is_ready(self):
        rows = self.volume_rows()
        del rows[4]
        summary = self.summary(rows)
        self.assertEqual(summary["windows"][1]["status"], "READY")
        self.assertEqual(summary["relative_volume"]["status"], "GAPS")
        self.assertEqual(summary["relative_volume"]["gap_count"], 1)
        self.assertEqual(summary["relative_volume"]["current_volume"], "35.000000")
        self.assertIsNone(summary["relative_volume"]["ratio"])

    def test_zero_current_and_zero_baseline_are_different_states(self):
        rows = self.volume_rows(current=0)
        volume = self.summary(rows)["relative_volume"]
        self.assertEqual(volume["status"], "READY")
        self.assertEqual(volume["ratio"], "0.000000")
        for row in rows[:-5]:
            row["volume"] = "0"
        for current in (0, 1):
            for row in rows[-5:]:
                row["volume"] = str(current)
            volume = self.summary(rows)["relative_volume"]
            self.assertEqual(volume["status"], "ZERO_BASELINE")
            self.assertEqual(volume["baseline_mean_volume"], "0.000000")
            self.assertIsNone(volume["ratio"])

    def test_stale_ratio_is_not_exposed(self):
        volume = self.summary(self.volume_rows(), self.now+timedelta(seconds=181))["relative_volume"]
        self.assertEqual(volume["status"], "STALE")
        self.assertIsNone(volume["current_volume"])
        self.assertIsNone(volume["baseline_mean_volume"])
        self.assertIsNone(volume["ratio"])

    def test_tiny_positive_volume_uses_unrounded_values_for_ratio(self):
        rows = self.volume_rows()
        for row in rows[:-5]:
            row["volume"] = "0.00000001"
        for row in rows[-5:]:
            row["volume"] = "0.00000002"
        summary = self.summary(rows)
        volume = summary["relative_volume"]
        self.assertEqual(volume["status"], "READY")
        self.assertEqual(volume["ratio"], "2.000000")
        self.assertGreater(D(volume["current_volume"]), 0)
        self.assertGreater(D(volume["baseline_mean_volume"]), 0)
        self.assertEqual(summary["windows"][1]["volume"],volume["current_volume"])
        self.assertGreater(D(summary["windows"][1]["mean_volume"]),0)


if __name__ == "__main__":
    unittest.main()
