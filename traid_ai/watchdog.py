"""Независимый контроль прогресса; callback только сигнализирует владельцу."""

from datetime import datetime
import math
import threading
import time

from .models import stamp, utc_now


class Watchdog:
    def __init__(self, on_trip, *, stale_seconds=45, resume_gap_seconds=30,
                 clock=time.monotonic, wall_clock=utc_now):
        for value in (stale_seconds, resume_gap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Порог watchdog должен быть положительным и конечным")
        self._on_trip = on_trip
        self._stale_seconds = stale_seconds
        self._resume_gap_seconds = resume_gap_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = "IDLE"
        self._last_progress_at = None
        self._last_progress_mono = None
        self._last_progress_wall = None
        self._last_check_mono = None
        self._last_check_wall = None
        self._tripped = False
        self._reason = None
        self._detected_at = None
        self._callback_error = None

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            mono, wall = self._clock(), self._wall_clock()
            self._stop.clear()
            self._status = "HEALTHY"
            self._last_progress_at = None
            self._last_progress_mono = self._last_check_mono = mono
            self._last_progress_wall = self._last_check_wall = wall
            self._tripped = False
            self._reason = self._detected_at = self._callback_error = None
            self._thread = threading.Thread(target=self._run, name="traid-watchdog", daemon=True)
            self._thread.start()

    def progress(self, now=None):
        """Вызывать только после завершённого шага владельца исполнения.

        datetime задаёт отметку UTC; число задаёт monotonic для тестирования.
        Уже обнаруженный сбой и запрет новых входов не снимаются.
        """
        with self._lock:
            if self._status in {"IDLE", "STOPPED"}:
                return
            mono = self._clock() if now is None or isinstance(now, datetime) else now
            wall = now if isinstance(now, datetime) else self._wall_clock()
            self._last_progress_at = stamp(wall)
            self._last_progress_mono = mono
            self._last_progress_wall = wall

    def _view_locked(self, mono, wall):
        age = None
        if self._last_progress_mono is not None:
            age = max(0, mono - self._last_progress_mono,
                      (wall - self._last_progress_wall).total_seconds())
        return {"status": self._status, "last_progress_at": self._last_progress_at,
                "age_seconds": age, "reason": self._reason, "tripped": self._tripped,
                "detected_at": self._detected_at, "callback_error": self._callback_error}

    def view(self):
        with self._lock:
            return self._view_locked(self._clock(), self._wall_clock())

    def check(self):
        reason = None
        with self._lock:
            mono, wall = self._clock(), self._wall_clock()
            if self._status not in {"IDLE", "STOPPED"} and not self._tripped:
                mono_gap = mono - self._last_check_mono
                wall_gap = (wall - self._last_check_wall).total_seconds()
                age = max(mono - self._last_progress_mono,
                          (wall - self._last_progress_wall).total_seconds())
                if (max(mono_gap, wall_gap) > self._resume_gap_seconds
                        or abs(wall_gap - mono_gap) > self._resume_gap_seconds):
                    self._status, reason = "RESUME_GAP", "SYSTEM_RESUME_GAP"
                elif age > self._stale_seconds:
                    self._status, reason = "STALLED", "WORKER_STALLED"
                self._last_check_mono, self._last_check_wall = mono, wall
                if reason is not None:
                    self._tripped = True
                    self._reason = reason
                    self._detected_at = stamp(wall)
        # Callback не получает executor, broker, ключи или БД; не держим lock.
        if reason is not None:
            try:
                self._on_trip(reason)
            except Exception:
                with self._lock:
                    self._callback_error = "CALLBACK_FAILED"
        return self.view()

    def _run(self):
        interval = min(1.0, self._stale_seconds / 4, self._resume_gap_seconds / 4)
        while not self._stop.wait(interval):
            self.check()

    def stop(self):
        with self._lock:
            self._stop.set()
            self._status = "STOPPED"
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
