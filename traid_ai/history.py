"""Closed minute candles and read-only background acquisition; no trading access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import threading
import urllib.request

from .models import D, decimal, encode, parse_time, stamp, utc_now

MAX_CANDLES = 240
HORIZONS = (1, 5, 15, 60)
VERSION = "candle-analytics-v2"
VOLUME_WINDOW_MINUTES = 5
VOLUME_BASELINE_WINDOWS = 6
KRAKEN_URL = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
ERROR_CODE = "CANDLE_DATA_UNAVAILABLE"


def _time(value):
    return parse_time(value) if isinstance(value, str) else parse_time(stamp(value))


def validate_candles(rows, source, now):
    """Fail closed on malformed/cross-source/future data; never fill missing bars."""
    now = _time(now)
    if not isinstance(rows, list) or (rows and source not in {"kraken-public", "alpaca-paper"}):
        raise ValueError("Некорректный источник свечей")
    normalized, previous = [], None
    try:
        for row in rows:
            if row["source"] != source or row["instrument"] != "BTC/USD":
                raise ValueError("Источник свечей не совпадает с сессией")
            start, end, available = (_time(row[key]) for key in ("start_at", "end_at", "available_at"))
            if (start.second or start.microsecond or end-start != timedelta(minutes=1)
                    or available < end or available > now or end > now
                    or (previous is not None and start <= previous)):
                raise ValueError("Некорректное время свечи")
            values = {key: decimal(row[key]) for key in ("open", "high", "low", "close", "volume")}
            if (min(values[key] for key in ("open", "high", "low", "close")) <= 0
                    or values["volume"] < 0 or values["low"] > min(values["open"], values["close"])
                    or values["high"] < max(values["open"], values["close"])
                    or values["high"] < values["low"]):
                raise ValueError("Некорректные OHLCV")
            normalized.append(dict(source=source, instrument="BTC/USD", start_at=stamp(start),
                                   end_at=stamp(end), available_at=stamp(available),
                                   **{key: str(value) for key, value in values.items()}))
            previous = start
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Повреждённые свечи") from exc
    return normalized[-MAX_CANDLES:]


def _row(source, start, now, opening, high, low, close, volume):
    return dict(source=source, instrument="BTC/USD", start_at=stamp(start),
                end_at=stamp(start+timedelta(minutes=1)), available_at=stamp(now),
                open=opening, high=high, low=low, close=close, volume=volume)


def parse_kraken_candles(payload, now):
    now = _time(now)
    if (not isinstance(payload, dict) or payload.get("error") != []
            or not isinstance(payload.get("result"), dict)
            or set(payload["result"]) != {"XXBTZUSD", "last"}):
        raise ValueError("Неожиданный ответ истории Kraken")
    bars = payload["result"]["XXBTZUSD"]
    if not isinstance(bars, list):
        raise ValueError("Повреждённая история Kraken")
    rows = []
    try:
        # Kraken always includes the final uncommitted timeframe, even with since.
        for bar in bars[:-1]:
            if not isinstance(bar, list) or len(bar) != 8:
                raise ValueError("Повреждённая свеча Kraken")
            timestamp = decimal(bar[0])
            if timestamp != timestamp.to_integral_value():
                raise ValueError("Некорректное время свечи Kraken")
            start = datetime.fromtimestamp(int(timestamp), timezone.utc)
            rows.append(_row("kraken-public", start, now, *bar[1:5], bar[6]))
    except (TypeError, OverflowError, OSError) as exc:
        raise ValueError("Повреждённая история Kraken") from exc
    return validate_candles(rows, "kraken-public", now)


def parse_alpaca_candles(payload, now):
    now = _time(now)
    if (not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict)
            or set(payload["bars"]) != {"BTC/USD"}
            or not isinstance(payload["bars"]["BTC/USD"], list)):
        raise ValueError("Неожиданный ответ истории Alpaca")
    rows = []
    try:
        for bar in payload["bars"]["BTC/USD"]:
            start = _time(bar["t"])
            if start+timedelta(minutes=1) > now:
                continue  # Current/incomplete bars must not enter decision context.
            rows.append(_row("alpaca-paper", start, now, *(bar[key] for key in ("o", "h", "l", "c", "v"))))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Повреждённая история Alpaca") from exc
    return validate_candles(rows, "alpaca-paper", now)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Редиректы истории запрещены")


def load_kraken_candles(now=None):
    """Fixed public GET; no credentials, environment proxies or redirects."""
    request = urllib.request.Request(KRAKEN_URL, headers={"User-Agent": "TraidAI-Paper/0.5", "Cache-Control": "no-cache"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=8) as response:
        raw = response.read(1024*1024+1)
    if len(raw) > 1024*1024:
        raise ValueError("Слишком большой ответ истории")
    # Production availability is receipt time, never the request start time.
    received_at = utc_now()
    if now is not None:
        received_at = max(received_at, _time(now))
    return parse_kraken_candles(json.loads(raw, parse_float=str), received_at)


def _number(value):
    return format(value, ".6f")


def _volume_number(value):
    rounded = _number(value)
    return str(value) if value and D(rounded) == 0 else rounded


def _movement(closes, high, low, net_return, variation):
    """Descriptors of the observed close path, never a regime forecast."""
    steps = [abs(b-a) for a, b in zip(closes, closes[1:])]
    path = sum(steps, D(0))
    return dict(
        path_efficiency_pct=_number(abs(closes[-1]-closes[0])/path*100) if path else None,
        largest_step_share_pct=_number(max(steps)/path*100) if path else None,
        pullback_from_high_pct=_number((high-closes[-1])/high*100),
        close_position_in_range_pct=_number((closes[-1]-low)/(high-low)*100) if high > low else None,
        return_to_variation_ratio=_number(net_return/variation) if variation else None)


def _window(rows, minutes, stale):
    selected = rows[-minutes:]
    gap_count = sum(_time(b["start_at"]) != _time(a["end_at"]) for a, b in zip(selected, selected[1:]))
    quality = "UNAVAILABLE" if not selected else "STALE" if stale else "GAPS" if gap_count else "WARMUP" if len(selected) < minutes else "READY"
    result = dict(horizon_minutes=minutes, horizon_seconds=minutes*60, status=quality,
                  quality=quality, count=len(selected), gap_count=gap_count,
                  return_pct=None, range_pct=None, volume=None, sampled_variation_pct=None,
                  mean_volume=None, volume_ratio_last_to_prior=None, movement=None)
    if quality != "READY":
        return result
    opening = D(selected[0]["open"])
    closes = [opening]+[D(row["close"]) for row in selected]
    returns = [(b/a-1)*100 for a, b in zip(closes, closes[1:])]
    net_return = (closes[-1]/opening-1)*100
    variation = sum((r*r for r in returns), D(0)).sqrt()
    high, low = max(D(r["high"]) for r in selected), min(D(r["low"]) for r in selected)
    volumes = [D(row["volume"]) for row in selected]
    prior_mean = sum(volumes[:-1], D(0))/(len(volumes)-1) if len(volumes) > 1 else None
    result.update(return_pct=_number(net_return),
                  range_pct=_number((high-low)/opening*100),
                  volume=_volume_number(sum(volumes, D(0))), mean_volume=_volume_number(sum(volumes, D(0))/len(volumes)),
                  sampled_variation_pct=_number(variation),
                  volume_ratio_last_to_prior=_number(volumes[-1]/prior_mean) if prior_mean else None,
                  movement=_movement(closes, high, low, net_return, variation))
    return result


def _relative_volume(rows, current, stale):
    """Compare equal, disjoint windows; require the entire prior baseline."""
    minutes, baseline_windows = VOLUME_WINDOW_MINUTES, VOLUME_BASELINE_WINDOWS
    required = minutes*(baseline_windows+1)
    selected = rows[-required:]
    gaps = sum(_time(b["start_at"]) != _time(a["end_at"]) for a, b in zip(selected, selected[1:]))
    status = ("UNAVAILABLE" if not selected else "STALE" if stale else "GAPS" if gaps else
              "WARMUP" if len(selected) < required else "READY")
    volume = sum((D(r["volume"]) for r in rows[-minutes:]), D(0)) if current["status"] == "READY" else None
    result = dict(status=status, window_minutes=minutes, baseline_windows=baseline_windows,
                  required_candles=required, observed_candles=len(selected), gap_count=gaps,
                  start_at=selected[0]["start_at"] if selected else None,
                  end_at=selected[-1]["end_at"] if selected else None,
                  current_volume=_volume_number(volume) if volume is not None else None,
                  baseline_mean_volume=None, ratio=None)
    if status != "READY":
        return result
    prior_volume = sum((D(r["volume"]) for r in selected[:-minutes]), D(0))
    mean = prior_volume/baseline_windows
    result["baseline_mean_volume"] = _volume_number(mean)
    if mean == 0:
        result["status"] = "ZERO_BASELINE"
    else:
        # Use original precision, not the rounded display of current_volume.
        result["ratio"] = _number(volume/mean)
    return result


def summarize_candles(rows, source, now):
    now = _time(now)
    rows = validate_candles(rows, source, now)
    latest = rows[-1]["end_at"] if rows else None
    age = (now-_time(latest)).total_seconds() if latest else None
    stale = age is not None and age > 180
    windows = [_window(rows, minutes, stale) for minutes in HORIZONS]
    status = ("UNAVAILABLE" if not rows else "STALE" if stale else
              "GAPS" if any(w["status"] == "GAPS" for w in windows) else
              "WARMUP" if len(rows) < max(HORIZONS) else "READY")
    return dict(version=VERSION, status=status, source=source, instrument="BTC/USD", count=len(rows),
                latest_end_at=latest, age_seconds=age, volume_available=bool(rows), windows=windows,
                relative_volume=_relative_volume(rows, next(w for w in windows if w["horizon_minutes"] == 5), stale),
                provenance=dict(source=source, instrument="BTC/USD",
                                available_at=max((r["available_at"] for r in rows), default=None),
                                candle_set_sha256=hashlib.sha256(encode(rows).encode()).hexdigest()))


class CandleFeed:
    """One background data-only loader; refresh and view never await network I/O."""
    def __init__(self, loader, source):
        self.loader, self.source = loader, source
        self._lock = threading.Lock()
        self._rows, self._error, self._last_attempt = [], None, None
        self._running = self._closed = False

    def refresh(self, now):
        now = _time(now)
        with self._lock:
            if (self._closed or self._running or (self._last_attempt is not None
                    and (now-self._last_attempt).total_seconds() < 60)):
                return False
            self._last_attempt, self._running = now, True
        threading.Thread(target=self._run, args=(now,), daemon=True, name="candle-feed").start()
        return True

    def _run(self, now):
        try:
            rows = self.loader(now)
            # A live loader records receipt after request start; validate at receipt.
            validated_at = max(now, utc_now())
            rows = validate_candles(rows, self.source, validated_at)
            error = None
        except Exception:
            rows, error = None, ERROR_CODE
        with self._lock:
            self._running = False
            if not self._closed:
                if rows is not None:
                    self._rows = rows
                self._error = error

    def view(self, now):
        now = _time(now)
        with self._lock:
            # Rows fetched after this decision time are never visible in replay.
            rows = deepcopy([r for r in self._rows if _time(r["available_at"]) <= now])
            error, running = self._error, self._running
        summary = summarize_candles(rows, self.source, now)
        summary.update(error_code=error, refreshing=running)
        return dict(rows=rows, summary=summary)

    def close(self):
        with self._lock:
            self._closed = True
