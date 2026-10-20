"""Data requests in background; the owning engine remains the only executor."""
import threading
import time
import math
from .models import stamp, utc_now


class QuoteFeed:
    def __init__(self, loader, *, interval_seconds=10, clock=time.monotonic):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Интервал котировок должен быть положительным")
        self.loader = loader
        self.interval_seconds, self.clock, self._last_attempt = interval_seconds, clock, None
        self._lock = threading.Lock()
        self._running = self._closed = False
        self._result = None
        self._started_at = self._success_at = None
        self._duration = None
        self._error = None

    def refresh(self):
        with self._lock:
            if self._closed or self._running or self._result is not None:
                return False
            now = self.clock()
            if self._last_attempt is not None and now-self._last_attempt < self.interval_seconds:
                return False
            self._last_attempt = now
            self._running = True
            self._started_at = stamp(utc_now())
        threading.Thread(target=self._run, daemon=True, name="quote-feed").start()
        return True

    def _run(self):
        started = self.clock()
        try:
            result = (self.loader(), None)
        except Exception:
            result = (None, "Источник котировок недоступен")
        with self._lock:
            self._running = False
            if not self._closed:
                self._result = result
                self._duration = round(max(0, self.clock()-started), 3)
                self._error = result[1]
                if result[1] is None:
                    self._success_at = stamp(utc_now())

    def view(self):
        with self._lock:
            return dict(status="STOPPED" if self._closed else "READING" if self._running else
                        "ERROR" if self._error else "READY" if self._success_at else "WAITING",
                        started_at=self._started_at, last_success_at=self._success_at,
                        duration_seconds=self._duration, error=self._error)

    def take(self):
        with self._lock:
            result, self._result = self._result, None
            return result if result is not None else (None, None)

    def close(self):
        with self._lock:
            self._closed = True
            self._result = None
