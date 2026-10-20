from copy import deepcopy
from datetime import datetime, timedelta, timezone
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from traid_ai.history import (CandleFeed, ERROR_CODE, KRAKEN_URL, _NoRedirect,
                              load_kraken_candles, parse_alpaca_candles,
                              parse_kraken_candles, summarize_candles, validate_candles)
from traid_ai.models import stamp


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)

    def rows(self, count=60, source="kraken-public"):
        result = []
        for index in range(count):
            start = self.now-timedelta(minutes=count-index)
            result.append(dict(source=source, instrument="BTC/USD", start_at=stamp(start),
                               end_at=stamp(start+timedelta(minutes=1)), available_at=stamp(self.now),
                               open="100", high="102", low="99", close="101", volume=str(index+1)))
        return result

    def test_summary_consecutive_closed_history_is_ready(self):
        result = summarize_candles(self.rows(), "kraken-public", self.now)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["age_seconds"], 0)
        self.assertEqual([w["horizon_minutes"] for w in result["windows"]], [1, 5, 15, 60])
        minute = result["windows"][0]
        self.assertEqual(minute["return_pct"], "1.000000")
        self.assertEqual(minute["range_pct"], "3.000000")
        self.assertEqual(minute["sampled_variation_pct"], "1.000000")
        self.assertEqual(minute["volume"], "60.000000")
        self.assertIsNone(minute["volume_ratio_last_to_prior"])
        self.assertEqual(result["windows"][-1]["mean_volume"], "30.500000")
        self.assertEqual(len(result["provenance"]["candle_set_sha256"]), 64)

    def test_unavailable_warmup_and_stale_do_not_invent_values(self):
        missing = summarize_candles([], "synthetic", self.now)
        self.assertEqual(missing["status"], "UNAVAILABLE")
        self.assertFalse(missing["volume_available"])
        warmup = summarize_candles(self.rows(1), "kraken-public", self.now)
        self.assertEqual(warmup["status"], "WARMUP")
        self.assertIsNone(warmup["windows"][1]["return_pct"])
        stale = summarize_candles(self.rows(), "kraken-public", self.now+timedelta(seconds=181))
        self.assertEqual(stale["status"], "STALE")
        self.assertIsNone(stale["windows"][0]["volume"])
        self.assertEqual(summarize_candles(self.rows(), "kraken-public", self.now+timedelta(seconds=180))["status"], "READY")

    def test_gap_is_preserved_and_invalidates_affected_window_only(self):
        rows = self.rows()
        del rows[-3]
        result = summarize_candles(rows, "kraken-public", self.now)
        self.assertEqual(result["status"], "GAPS")
        self.assertEqual(result["count"], 59)
        self.assertEqual(result["windows"][0]["status"], "READY")
        for window in result["windows"][1:]:
            self.assertEqual(window["status"], "GAPS")
            self.assertIsNone(window["return_pct"])
            self.assertIsNone(window["volume"])

    def test_rejects_future_source_duplicates_order_and_interval(self):
        for field, value in (("source", "alpaca-paper"), ("instrument", "ETH/USD"),
                             ("available_at", stamp(self.now+timedelta(seconds=1))),
                             ("available_at", stamp(self.now-timedelta(minutes=2))),
                             ("end_at", stamp(self.now+timedelta(seconds=1))),
                             ("start_at", stamp(self.now-timedelta(seconds=59)))):
            with self.subTest(field=field, value=value):
                rows = self.rows(1)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate_candles(rows, "kraken-public", self.now)
        rows = self.rows(2)
        for invalid in ([rows[0], rows[0]], list(reversed(rows))):
            with self.assertRaises(ValueError):
                validate_candles(invalid, "kraken-public", self.now)

    def test_rejects_nonfinite_negative_inconsistent_and_missing_values(self):
        for field, value in (("open", "NaN"), ("volume", "Infinity"), ("volume", "-1"),
                             ("close", "0"), ("low", "101"), ("high", "99"), ("open", True)):
            with self.subTest(field=field, value=value):
                rows = self.rows(1)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate_candles(rows, "kraken-public", self.now)
        rows = self.rows(1)
        del rows[0]["volume"]
        with self.assertRaises(ValueError):
            validate_candles(rows, "kraken-public", self.now)

    def test_normalization_retains_latest_240_and_provenance_changes(self):
        rows = self.rows(250)
        original = deepcopy(rows)
        normalized = validate_candles(rows, "kraken-public", self.now)
        self.assertEqual(len(normalized), 240)
        self.assertEqual(normalized[0], rows[10])
        self.assertEqual(rows, original)
        first = summarize_candles(rows, "kraken-public", self.now)
        rows[-1]["volume"] = "5"
        second = summarize_candles(rows, "kraken-public", self.now)
        self.assertNotEqual(first["provenance"]["candle_set_sha256"], second["provenance"]["candle_set_sha256"])

    def kraken_payload(self):
        return {"error": [], "result": {"XXBTZUSD": [
            [int((self.now-timedelta(minutes=1)).timestamp()), "100", "102", "99", "101", "100", "2", 3],
            [int(self.now.timestamp()), "100", "102", "99", "101", "100", "999", 3]], "last": int(self.now.timestamp())}}

    def test_kraken_drops_last_uncommitted_row_even_if_locally_in_past(self):
        payload = self.kraken_payload()
        rows = parse_kraken_candles(payload, self.now+timedelta(minutes=5))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["volume"], "2")
        self.assertEqual(rows[0]["source"], "kraken-public")

    def test_kraken_rejects_malformed_error_instrument_and_future_closed_bar(self):
        payload = self.kraken_payload()
        payload["result"]["XXBTZUSD"][0][0] += 120
        with self.assertRaises(ValueError):
            parse_kraken_candles(payload, self.now)
        for payload in ({"error": ["bad"]}, {"error": [], "result": {"ETHUSD": [], "last": 0}},
                        {"error": [], "result": {"XXBTZUSD": [[1], []], "last": 0}}):
            with self.assertRaises(ValueError):
                parse_kraken_candles(payload, self.now)

    def test_alpaca_excludes_incomplete_and_preserves_missing_minutes(self):
        rows = self.rows(5, "alpaca-paper")
        bars = [dict(t=r["start_at"], o=r["open"], h=r["high"], l=r["low"], c=r["close"], v=r["volume"]) for r in rows]
        del bars[2]
        bars.append(dict(t=stamp(self.now), o=100, h=102, l=99, c=101, v=3))
        parsed = parse_alpaca_candles({"bars": {"BTC/USD": bars}}, self.now)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(summarize_candles(parsed, "alpaca-paper", self.now)["windows"][1]["status"], "GAPS")
        with self.assertRaises(ValueError):
            parse_alpaca_candles({"bars": {"ETH/USD": bars}}, self.now)

    def test_zero_prior_volume_ratio_is_unknown(self):
        rows = self.rows(5)
        for row in rows[:-1]:
            row["volume"] = "0"
        result = summarize_candles(rows, "kraken-public", self.now)
        self.assertIsNone(result["windows"][1]["volume_ratio_last_to_prior"])

    def test_loader_is_fixed_get_bounded_and_has_no_redirect_or_proxy(self):
        import json
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(self.kraken_payload()).encode()
        opener = MagicMock()
        opener.open.return_value = response
        with patch("traid_ai.history.urllib.request.build_opener", return_value=opener) as build:
            with patch("traid_ai.history.utc_now", return_value=self.now):
                rows = load_kraken_candles(self.now)
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], _NoRedirect)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, KRAKEN_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 8})
        response.__enter__.return_value.read.assert_called_once_with(1024*1024+1)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            _NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid")
        response.__enter__.return_value.read.return_value = b"x"*(1024*1024+1)
        with patch("traid_ai.history.urllib.request.build_opener", return_value=opener):
            with self.assertRaises(ValueError):
                load_kraken_candles(self.now)

    def test_background_is_nonblocking_single_flight_and_rate_limited(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        def loader(now):
            calls.append(now)
            started.set()
            release.wait(2)
            return self.rows()
        feed = CandleFeed(loader, "kraken-public")
        before = time.monotonic()
        self.assertTrue(feed.refresh(self.now))
        self.assertTrue(started.wait(1))
        self.assertEqual(feed.view(self.now)["summary"]["status"], "UNAVAILABLE")
        self.assertLess(time.monotonic()-before, 0.5)
        self.assertFalse(feed.refresh(self.now+timedelta(minutes=2)))
        release.set()
        self.wait_ready(feed)
        self.assertFalse(feed.refresh(self.now+timedelta(seconds=59)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(feed.refresh(self.now+timedelta(seconds=60)))
        feed.close()

    def wait_ready(self, feed):
        deadline = time.monotonic()+2
        while feed.view(self.now)["summary"]["refreshing"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(feed.view(self.now)["summary"]["refreshing"])

    def test_background_failure_retains_rows_and_sanitizes_errors(self):
        fail = [False]
        def loader(now):
            if fail[0]:
                raise RuntimeError("secret token=PRIVATE")
            return self.rows()
        feed = CandleFeed(loader, "kraken-public")
        feed.refresh(self.now)
        self.wait_ready(feed)
        fail[0] = True
        feed.refresh(self.now+timedelta(minutes=1))
        self.wait_ready(feed)
        view = feed.view(self.now+timedelta(minutes=1))
        self.assertEqual(len(view["rows"]), 60)
        self.assertEqual(view["summary"]["error_code"], ERROR_CODE)
        self.assertNotIn("PRIVATE", str(view))
        self.assertEqual(feed.view(self.now-timedelta(seconds=1))["rows"], [])
        feed.close()

    def test_close_invalidates_late_result(self):
        release, done = threading.Event(), threading.Event()
        def loader(now):
            release.wait(2)
            done.set()
            return self.rows()
        feed = CandleFeed(loader, "kraken-public")
        feed.refresh(self.now)
        feed.close()
        release.set()
        self.assertTrue(done.wait(1))
        self.wait_ready(feed)
        self.assertEqual(feed.view(self.now)["rows"], [])
        self.assertFalse(feed.refresh(self.now+timedelta(minutes=2)))


if __name__ == "__main__":
    unittest.main()
