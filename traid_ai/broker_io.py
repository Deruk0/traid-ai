"""One read-only reconciliation job; only the owner applies results or sends orders."""
from copy import deepcopy
from contextlib import nullcontext
import threading
import time

from .alpaca import AlpacaError
from .models import stamp, utc_now


class BrokerReadCycle:
    def __init__(self, client, *, budget_seconds=5, clock=time.monotonic):
        self.client = client.data_client() if hasattr(client, "data_client") else client
        self.budget = budget_seconds
        self.clock = clock
        self.lock = threading.Lock()
        self.job = None
        self.closed = False

    def _stage(self, job, stage):
        with self.lock:
            job["stage"] = stage

    def _run(self, job, client_id):
        result = {}
        try:
            budget = self.client.request_budget(self.budget) if hasattr(self.client, "request_budget") else nullcontext()
            with budget:
                calls = [("order", lambda: self.client.order_by_client_id(client_id))] if client_id else []
                calls += [("account", self.client.account), ("positions", self.client.positions),
                          ("open_orders", self.client.open_orders)]
                for name, loader in calls:
                    with self.lock:
                        if self.closed or job["expired"]:
                            raise AlpacaError("Чтение остановлено")
                    if self.clock()-job["started"] >= self.budget:
                        raise AlpacaError("Истёк бюджет сверки брокера")
                    self._stage(job, name)
                    result[name] = loader()
                if client_id and result["order"] is None:
                    with self.lock:
                        if self.closed or job["expired"]:
                            raise AlpacaError("Чтение остановлено")
                    if self.clock()-job["started"] >= self.budget:
                        raise AlpacaError("Истёк бюджет сверки брокера")
                    self._stage(job, "order_history")
                    result["history"] = self.client.orders()
            error = None
        except Exception:
            # Never expose arbitrary transport exceptions or credentials.
            error = "BROKER_READ_FAILED"
        with self.lock:
            job.update(result=result if error is None else None, error=error,
                       finished=self.clock())
            job["done"].set()

    def poll(self, client_id=None):
        with self.lock:
            if self.closed:
                raise AlpacaError("Сверка остановлена")
            job = self.job
            if job is None:
                job = dict(started=self.clock(), started_at=stamp(utc_now()), stage="starting",
                           done=threading.Event(), result=None, error=None, client_id=client_id,
                           expired=False)
                self.job = job
                threading.Thread(target=self._run, args=(job, client_id), daemon=True,
                                 name="broker-read-only").start()
        # Fast in-memory clients complete in this interval; a network cannot hold the owner.
        job["done"].wait(.02)
        with self.lock:
            elapsed = self.clock()-job["started"]
            expired = job["expired"] or elapsed > self.budget
            if expired:
                job["expired"] = True
            if job["done"].is_set():
                self.job = None
                if expired or job["client_id"] != client_id:
                    raise AlpacaError("Поздний результат сверки отброшен; входы заблокированы")
                if job["error"]:
                    raise AlpacaError("Не удалось прочитать состояние брокера; входы заблокированы")
                return deepcopy(job["result"])
            if expired:
                raise AlpacaError("Истёк бюджет сверки; ожидается завершение чтения, входы заблокированы")
            return None

    def view(self):
        with self.lock:
            job = self.job
            if not job:
                return dict(status="IDLE", stage=None, elapsed_seconds=0, budget_seconds=self.budget)
            elapsed = max(0, self.clock()-job["started"])
            return dict(status="TIMED_OUT" if job["expired"] or elapsed > self.budget else "READING",
                        stage=job["stage"], started_at=job["started_at"],
                        elapsed_seconds=round(elapsed, 3), budget_seconds=self.budget)

    def close(self):
        # An existing worker may finish GETs, but cannot apply state or send orders.
        with self.lock:
            self.closed = True
