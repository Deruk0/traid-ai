"""Local-only dashboard; one worker owns the existing paper engine."""
from datetime import timedelta
from contextlib import ExitStack, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import sqlite3
import threading
from urllib.parse import parse_qs, urlparse

from .engine import Engine
from .alpaca import PaperConnection
from .broker import BrokerEngine, account_lock_path
from .qwen import ModelConnection
from .feeds import public_quote, synthetic_quote
from .models import Config, D, Quote, decimal, encode, parse_time, stamp, utc_now
from .store import Store, writer_lock
from .history import CandleFeed, load_kraken_candles
from .runtime import QuoteFeed
from .watchdog import Watchdog
from .health import technical_health
from .diagnostics import IMPORTANT_EVENTS, blocking_log, decision_summary


ASSETS = Path(__file__).with_name("web")


def parse_settings(payload: dict) -> tuple[Config, float]:
    mode = payload.get("mode", "demo")
    if mode not in {"demo", "paper", "alpaca"}:
        raise ValueError("Выберите режим сессии")
    strategy = payload.get("strategy", "momentum")
    if strategy not in {"momentum", "smoke", "qwen"}:
        raise ValueError("Неизвестный алгоритм сессии")
    minutes = decimal(payload.get("duration_minutes", "60"))
    if not 1 < minutes <= 1440:
        raise ValueError("Длительность: больше 1 и до 1440 минут")
    hold = payload.get("max_hold_minutes")
    hold_seconds = decimal(hold)*60 if hold not in (None, "") else None
    if hold_seconds is not None and (hold_seconds <= 0 or hold_seconds != hold_seconds.to_integral_value()):
        raise ValueError("Максимум удержания должен задавать положительное целое число секунд")
    config = Config(source={"demo":"synthetic", "paper":"kraken-public", "alpaca":"alpaca-paper"}[mode],
                    capital=decimal(payload.get("capital", "1000")),
                    risk_per_trade=decimal(payload.get("risk_percent", "0.5")) / 100,
                    max_exposure=decimal(payload.get("exposure_percent", "25")) / 100,
                    loss_limit=decimal(payload.get("loss_limit_percent", "3")) / 100,
                    fee_rate=decimal(payload.get("fee_percent", "0.1")) / 100,
                    slippage_bps=decimal(payload.get("slippage_bps", "5")),
                    stop_fraction=decimal(payload.get("stop_percent","2"))/100,
                    target_return=decimal(payload["target_percent"])/100
                        if payload.get("target_percent") not in (None,"") else None,
                    llm_model=payload.get("llm_model", "qwen3.5:4b"),
                    max_hold_seconds=int(hold_seconds) if hold_seconds is not None else None)
    if strategy in {"smoke", "qwen"}:
        from dataclasses import replace
        config = replace(config, strategy_version="paper-smoke-v1" if strategy == "smoke" else "qwen-local-v1")
    if not D("5") <= config.capital <= D("1000000000"):
        raise ValueError("Виртуальный бюджет: от 5 до 1 000 000 000 USD")
    return config, float(minutes)


def chart_data(config: Config, state, events: list[dict]) -> dict:
    cash, qty, bid, ask = config.capital, D("0"), None, None
    points, fills = [], []
    group_time = None

    def point():
        if bid is not None:
            points.append(dict(time=group_time, price=str((bid + ask) / 2),
                               equity=str(cash + qty * bid)))

    for event in events:
        if group_time is not None and event["occurred_at"] != group_time:
            point()
        group_time = event["occurred_at"]
        payload = event["payload"]
        if event["kind"] == "SNAPSHOT":
            bid, ask = D(payload["bid"]), D(payload["ask"])
        elif event["kind"] == "BROKER_LEDGER":
            cash, qty = D(payload["cash"]), D(payload["quantity"])
        elif event["kind"] in {"PAPER_FILL", "BROKER_FILL"}:
            amount, price, fee = D(payload["quantity"]), D(payload["price"]), D(payload["fee"])
            if event["kind"] == "PAPER_FILL":
                if payload["side"] == "BUY":
                    cash -= amount * price + fee
                    qty += amount
                else:
                    cash += amount * price - fee
                    qty -= amount
            fills.append(dict(time=group_time, **payload))
    if group_time:
        point()
    if len(points) > 600:
        stride = math.ceil(len(points) / 599)
        points = points[::stride] + ([points[-1]] if points[-1] != points[::stride][-1] else [])
    return dict(points=points, fills=fills[-500:])


def decision_flow(events):
    """Link recorded identifiers; do not infer an execution from an AI approval."""
    rows, answers, owners = {}, {}, {}
    for event in events:
        p, kind = event["payload"], event["kind"]
        if kind == "AI_DECISION":
            answers[p["snapshot_id"]] = p
        decision = p.get("decision_id")
        if kind == "DECISION":
            ai = answers.get(p.get("ai_snapshot_id"))
            rows[decision] = dict(decision_id=decision, time=event["occurred_at"], action=p["action"],
                origin="QWEN" if ai or p.get("strategy_version") == "qwen-local-v1" else "CONTROL" if p.get("strategy_version") == "independent-control" else "RULE",
                reason=ai["reason"] if ai else p["reason"], risk=None, execution="NONE", filled_quantity="0")
            if ai and ai.get("explanation_status"):
                rows[decision].update(reason_code=ai.get("reason_code"),
                    explanation_status=ai["explanation_status"],evidence=ai.get("evidence",{}))
        if kind == "ORDER_PREPARED" and p.get("client_order_id"):
            owners[p["client_order_id"]] = decision
        if not decision:
            decision = owners.get(p.get("client_order_id")) or owners.get(p.get("order_id"))
        if decision not in rows:
            continue
        row = rows[decision]
        if kind == "RISK_VERDICT":
            row["risk"] = dict(approved=p["approved"], reason=p["reason"])
        elif kind in {"PAPER_FILL", "BROKER_FILL"}:
            row["filled_quantity"] = str(D(row["filled_quantity"])+D(p["quantity"]))
            row["execution"] = "FILLED" if kind == "PAPER_FILL" else "PARTIALLY_FILLED"
        elif kind == "ORDER_PREPARED":
            row["execution"] = "PREPARED"
        elif kind == "ORDER_UPDATE":
            row["execution"] = p["status"].upper()
            owners[p.get("order_id")] = decision
        elif kind in {"ORDER_UNKNOWN", "ORDER_REJECTED", "ORDER_ABORTED", "ORDER_CANCEL_REQUESTED"}:
            row["execution"] = kind.removeprefix("ORDER_")
    return list(rows.values())[-50:]


class SessionManager:
    def __init__(self, path: Path, *, demo_interval: float = 0.04):
        self.path = path.resolve()
        self.demo_interval = demo_interval
        self.thread = None
        self.active_id = None
        self.error = None
        self.mutex = threading.RLock()
        self.wake = threading.Event()
        self.stop_requested = False
        self.pause_requested = None
        self.alpaca = PaperConnection()
        self.execution_source = None
        self.model = ModelConnection()
        self.watchdog = Watchdog(self._watchdog_trip)
        self.watchdog_reason = None
        self.maintenance_thread = None
        self.maintenance = dict(running=False, kind=None, session_id=None, error=None)
        self.runtime = {}

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def launch(self, *, payload: dict | None = None, session_id: str | None = None):
        config, minutes = parse_settings(payload) if payload is not None else (None, None)
        with self.mutex:
            if self.running or self.maintenance["running"]:
                raise ValueError("Сессия уже работает. Завершите её перед новым запуском")
            if config and config.strategy_version == "qwen-local-v1":
                from dataclasses import replace
                model = self.model.view()
                if (not model["connected"] or model["model"] != config.llm_model
                        or model["info"]["model"] != config.llm_model):
                    raise ValueError("Сначала подключи выбранную модель Qwen и дождись успешной проверки")
                config = replace(config, llm_digest=model["info"]["digest"])
            ready = threading.Event()
            startup = {}
            self.stop_requested = False
            self.pause_requested = None
            self.wake.clear()
            self.error = None
            self.watchdog_reason = None
            self.runtime = {}
            self.execution_source = config.source if config else "RESTORE_PENDING"
            self.thread = threading.Thread(target=self._worker,
                                           args=(config, minutes, session_id, ready, startup),
                                           name="paper-session", daemon=True)
            self.thread.start()
        if not ready.wait(5):
            raise ValueError("Исполнитель запускается. Проверьте состояние перед повтором")
        if "error" in startup:
            raise ValueError(startup["error"])
        return startup["id"]

    def model_change(self, payload):
        with self.mutex:
            if self.running:
                raise ValueError("Сначала заверши текущую сессию; её модель зафиксирована")
            return self.model.connect(payload.get("model", "qwen3.5:4b"))

    def control(self, action: str, session_id: str):
        with self.mutex:
            if not self.running or session_id != self.active_id:
                raise ValueError("Выбранная сессия сейчас не выполняется")
            if action == "stop":
                self.stop_requested = True
            elif action in {"pause", "continue"}:
                self.pause_requested = action == "pause"
            else:
                raise ValueError("Неизвестная команда")
            self.wake.set()

    def _worker(self, config, minutes, session_id, ready, startup):
        store = None
        engine = None
        candle_feed = quote_feed = None
        try:
            with ExitStack() as ownership:
                ownership.enter_context(writer_lock(self.path))
                store = Store(self.path)
                target_config = store.load(session_id)[0] if session_id else config
                if target_config.source != "synthetic":
                    self.watchdog.start()
                client = None
                if target_config.source == "alpaca-paper":
                    client = self.alpaca.execution_client()
                    with client.request_budget(5) if hasattr(client, "request_budget") else nullcontext():
                        account = client.account()
                    ownership.enter_context(writer_lock(account_lock_path(account["account_id"])))
                if session_id:
                    if store.analysis(session_id,"experiment_role"):
                        raise ValueError("Исследовательский прогон нельзя продолжить как обычную сессию")
                    stored_config, _ = store.load(session_id)
                    engine = (BrokerEngine(store, session_id, client)
                              if stored_config.source == "alpaca-paper" else Engine(store, session_id))
                    if engine.state.status == "CLOSED":
                        raise ValueError("Эта сессия уже закрыта")
                else:
                    # Prevent abandoning an unfinished session when creating another.
                    if any(json.loads(row[0])["status"] != "CLOSED"
                           for row in store.db.execute("SELECT state FROM sessions")):
                        raise ValueError("Есть незавершённая сессия. Выберите её в истории и восстановите")
                    now = utc_now()
                    with client.request_budget(5) if hasattr(client, "request_budget") else nullcontext():
                        engine = (BrokerEngine.start_paper(store, config, now, now + timedelta(minutes=minutes), client)
                                  if config.source == "alpaca-paper" else
                                  Engine.start(store, config, now, now + timedelta(minutes=minutes)))
                self.execution_source = engine.config.source
                if hasattr(engine.strategy, "wake"):
                    engine.strategy.wake = self.wake
                if isinstance(engine, BrokerEngine):
                    engine.control_state = self._latest_control
                self.active_id = engine.state.session_id
                if engine.config.source != "synthetic":
                    self.watchdog.start()
                    if engine.config.source == "alpaca-paper":
                        data_client = client.data_client() if hasattr(client, "data_client") else client
                        def load_quote():
                            observed = data_client.quote()
                            return Quote("alpaca:"+observed["observed_at"], observed["observed_at"],
                                         D(observed["bid"]), D(observed["ask"]), "alpaca-paper")
                        quote_feed = QuoteFeed(load_quote)
                        if hasattr(data_client, "candles"):
                            candle_client = data_client.data_client() if hasattr(data_client, "data_client") else data_client
                            candle_feed = CandleFeed(candle_client.candles, "alpaca-paper")
                    else:
                        quote_feed = QuoteFeed(public_quote)
                        candle_feed = CandleFeed(load_kraken_candles, "kraken-public")
                    quote_feed.refresh()
                    if candle_feed and engine.config.strategy_version == "qwen-local-v1":
                        engine.candle_provider = candle_feed.view
                        candle_feed.refresh(utc_now())
                startup["id"] = self.active_id
                with self.mutex:
                    self.runtime = dict(session_id=self.active_id, quotes=quote_feed,
                                        broker=engine.read_cycle if isinstance(engine, BrokerEngine) else None)
                ready.set()
                start = parse_time(engine.state.started_at)
                deadline = parse_time(engine.state.deadline_at)
                duration = (deadline - start).total_seconds()
                samples = max(60, math.ceil(duration / 10))
                last = parse_time(engine.state.last_tick_at) if engine.state.last_tick_at else None
                index = math.floor((last - start).total_seconds() / duration * samples) + 1 if last else 0
                while engine.state.status != "CLOSED":
                    with self.mutex:
                        stop, pause = self.stop_requested, self.pause_requested
                        self.pause_requested = None
                        self.wake.clear()
                    quote, error = None, None
                    if engine.config.source == "synthetic":
                        now = start + timedelta(seconds=duration * min(index, samples) / samples)
                        # A stopped/restored model can end at its original last tick.
                        if last and now <= last:
                            now = last + timedelta(microseconds=1)
                        quote = synthetic_quote(index, now)
                        index += 1
                    else:
                        quote, error = quote_feed.take()
                        quote_feed.refresh()
                        now = utc_now()
                        if candle_feed and engine.config.strategy_version == "qwen-local-v1":
                            candle_feed.refresh(now)
                    # Commands arriving while the data request was in flight apply this tick.
                    with self.mutex:
                        stop = stop or self.stop_requested
                        if self.pause_requested is not None:
                            pause = self.pause_requested
                            self.pause_requested = None
                        watchdog_reason = self.watchdog_reason
                    if watchdog_reason and engine.state.exit_reason not in {"WORKER_STALLED", "SYSTEM_RESUME_GAP"}:
                        engine.config, engine.state = store.load(engine.state.session_id)
                        engine.state.exit_reason = watchdog_reason
                        store.commit_step(engine.state, secrets.token_hex(16), stamp(now),
                                          [("WATCHDOG_TRIPPED", {"reason": watchdog_reason})])
                    engine.step(now, quote, stop=stop, error=error, pause_entries=pause)
                    self.watchdog.progress()
                    last = now
                    if engine.state.status == "CLOSED" or (engine.state.status == "UNRESOLVED"
                            and (engine.config.source != "alpaca-paper" or stop)):
                        break
                    if engine.config.source == "synthetic":
                        delay = self.demo_interval
                    else:
                        flatten_at = deadline - timedelta(seconds=engine.config.flatten_buffer_seconds)
                        wake_at = flatten_at if now < flatten_at else deadline
                        if engine.config.source == "alpaca-paper":
                            reading = engine.read_cycle.view()["status"] != "IDLE"
                            delay = .25 if reading else 2 if engine.state.broker["pending"] else 10
                            if now < wake_at:
                                delay = min(delay, max(.1, (wake_at - now).total_seconds()))
                        else:
                            delay = min(10, max(0, (wake_at - now).total_seconds()))
                    self.wake.wait(delay)
        except Exception as exc:
            self.error = str(exc)
            startup["error"] = str(exc)
            if engine and store and engine.state.status != "CLOSED":
                try:
                    # ExitStack has released ownership: reacquire before any recovery write.
                    with ExitStack() as recovery:
                        recovery.enter_context(writer_lock(self.path))
                        engine.config, engine.state = store.load(engine.state.session_id)
                        if engine.config.source == "alpaca-paper":
                            recovery.enter_context(writer_lock(account_lock_path(engine.state.broker["account_id"])))
                        if engine.state.status != "CLOSED":
                            engine.state.status, engine.state.exit_reason = "UNRESOLVED", "WORKER_ERROR"
                            store.commit_step(engine.state, secrets.token_hex(16), stamp(utc_now()),
                                              [("WORKER_ERROR", {"reason": "WORKER_ERROR"})])
                except Exception:
                    pass
        finally:
            self.watchdog.stop()
            if quote_feed:
                quote_feed.close()
            if candle_feed:
                candle_feed.close()
            if engine and hasattr(engine.strategy, "close"):
                engine.strategy.close()
            if isinstance(engine, BrokerEngine):
                engine.close()
            with self.mutex:
                self.runtime = {}
            if store:
                store.close()
            ready.set()

    def _watchdog_trip(self, reason):
        with self.mutex:
            self.watchdog_reason = reason
            self.stop_requested = True
            self.wake.set()

    def delete_session(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Выберите сессию")
        with self.mutex:
            if self.running or self.maintenance["running"]:
                raise ValueError("Сначала завершите исполнение или текущую проверку")
            with writer_lock(self.path):
                store = Store(self.path)
                try:
                    store.delete_session(session_id)
                finally:
                    store.close()
            if self.active_id == session_id:
                self.active_id = None
                self.error = None
            return dict(deleted_id=session_id)

    def analyze_session(self, session_id, kind):
        if kind not in {"comparison", "accounting"}:
            raise ValueError("Неизвестная проверка")
        with self.mutex:
            if self.running or self.maintenance["running"]:
                raise ValueError("Сначала завершите исполнение или текущую проверку")
            view = self.view(session_id, full_events=True)
            allowed = {"CLOSED","UNRESOLVED"} if kind == "accounting" else {"CLOSED"}
            if not view["state"] or view["state"]["status"] not in allowed:
                raise ValueError("Сравнение требует закрытой сессии; сверка также доступна для UNRESOLVED")
            client = None
            if kind == "accounting":
                if view["config"]["source"] != "alpaca-paper":
                    raise ValueError("Внешние начисления есть только у Alpaca Paper")
                client = self.alpaca.execution_client()
            self.maintenance = dict(running=True, kind=kind, session_id=session_id, error=None)
            self.maintenance_thread = threading.Thread(target=self._analyze_worker,
                args=(session_id,kind,view,client), daemon=True, name="session-analysis")
            self.maintenance_thread.start()
            return dict(self.maintenance)

    def _analyze_worker(self, session_id, kind, view, client):
        store = None
        try:
            with ExitStack() as ownership:
                ownership.enter_context(writer_lock(self.path))
                store = Store(self.path)
                config, state = store.load(session_id)
                allowed = {"CLOSED","UNRESOLVED"} if kind == "accounting" else {"CLOSED"}
                if state.status not in allowed:
                    raise ValueError("Состояние сессии изменилось")
                if kind == "comparison":
                    from .evaluation import compare_session
                    report = compare_session(view)
                    store.save_analysis(session_id, kind, report)
                else:
                    from .accounting import reconcile_activities
                    account = client.account()
                    ownership.enter_context(writer_lock(account_lock_path(account["account_id"])))
                    now = utc_now()
                    start = parse_time(state.started_at).replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(seconds=1)
                    activities = client.activities(stamp(start), stamp(now))
                    account = client.account()
                    report = reconcile_activities(config,state,store.events(session_id),activities,now,
                        account=account, positions=client.positions(), open_orders=client.open_orders())
                    state.accounting = report
                    if report["pnl_is_final"]:
                        state.fees = report["fees_usd"]
                        state.cash = report["cash"]
                        state.realized_pnl = report["net_pnl"]
                    store.commit_step(state,secrets.token_hex(16),stamp(now),[("ACCOUNTING_RECONCILED",report)])
        except Exception:
            with self.mutex:
                self.maintenance["error"] = "Проверка не завершена; проверь подключение и журнал"
        finally:
            if store:
                store.close()
            with self.mutex:
                self.maintenance["running"] = False

    def _latest_control(self):
        self.watchdog.check()
        with self.mutex:
            return self.stop_requested, self.pause_requested

    def view(self, selected: str | None = None, *, full_events: bool = False) -> dict:
        result = dict(running=self.running, active_id=self.active_id, worker_error=self.error,
                      sessions=[], selected_id=None, report=None, config=None, state=None,
                      points=[], fills=[], events=[], important_events=[], blocking_events=[], event_count=0,
                      decision_summary=decision_summary([]))
        result["alpaca_connected"] = self.alpaca.view()["connected"]
        result["execution_source"] = self.execution_source
        result["capabilities"] = ["alpaca-paper-execution-v1"]
        result["capabilities"].append("qwen-local-decisions-v1")
        result["model"] = self.model.view()
        result["watchdog"] = self.watchdog.view()
        result["maintenance"] = dict(self.maintenance)
        result["capabilities"].extend(["candle-context-v1", "watchdog-v1", "comparison-v1", "accounting-v1", "decision-evidence-v1", "technical-health-v1", "bounded-broker-reads-v1"])
        result["capabilities"].append("session-diagnostics-v1")
        result["capabilities"].append("wait-review-v1")
        result["capabilities"].append("trailing-objective-v1")
        result["capabilities"].append("qwen-thinking-v1")
        result["capabilities"].append("qwen-model-selection-v2")
        result["capabilities"].append("broker-decision-handoff-v1")
        result["capabilities"].append("panel-usability-v1")
        result["capabilities"].append("qwen-result-delivery-v1")
        result["capabilities"].append("execution-safety-guards-v1")
        result["capabilities"].append("optional-max-hold-v1")
        result["capabilities"].append("terminal-balance-reconciliation-v1")
        result["capabilities"].append("trade-blocking-log-v1")
        result["health"] = technical_health(None, None, [], utc_now())
        if not self.path.exists():
            return result
        store = Store(self.path, readonly=True)
        try:
            store.db.execute("BEGIN")
            result["sessions"] = store.sessions()
            active_exists = any(s["session_id"] == self.active_id for s in result["sessions"])
            selected = selected or (self.active_id if active_exists else None) or (result["sessions"][0]["session_id"] if result["sessions"] else None)
            if not selected:
                return result
            config, _ = store.load(selected)
            engine = BrokerEngine(store, selected) if config.source == "alpaca-paper" else Engine(store, selected)
            events = store.events(selected)
            result["important_events"] = [e for e in events if e["kind"] in IMPORTANT_EVENTS][-50:]
            result["decision_summary"] = decision_summary(events)
            result["blocking_events"] = blocking_log(events, limit=None if full_events else 200)
            result.update(selected_id=selected, report=engine.report(), config=engine.config.to_dict(),
                          state=engine.state.to_dict(), event_count=len(events),
                          events=events if full_events else events[-500:],
                          **chart_data(engine.config, engine.state, events))
            result["comparison"] = store.analysis(selected, "comparison")
            result["decision_flow"] = decision_flow(events)
            active = self.running and selected == self.active_id
            with self.mutex:
                runtime = dict(self.runtime) if self.runtime.get("session_id") == selected else {}
            result["health"] = technical_health(engine.config, engine.state, events, utc_now(),
                active=active, watchdog=result["watchdog"],
                quote_io=runtime["quotes"].view() if runtime.get("quotes") else None,
                broker_io=runtime["broker"].view() if runtime.get("broker") else None)
            return result
        finally:
            store.close()

    def shutdown(self):
        with self.mutex:
            if self.running:
                self.stop_requested = True
                self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)
        self.watchdog.stop()
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=2)
        self.alpaca.shutdown()

    def connection_change(self, *, payload=None):
        with self.mutex:
            if self.running and self.execution_source not in {"synthetic", "kraken-public"}:
                raise ValueError("Сначала завершите сессию Alpaca; исполнитель ещё управляет её заявками")
            if payload is None:
                self.alpaca.disconnect()
                return self.alpaca.view()
            return self.alpaca.connect(payload.get("key"), payload.get("secret"))


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        self.manager = manager
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _host_allowed(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _reply(self, status: int, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else encode(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._host_allowed():
            self._reply(403, {"error": "Недопустимый адрес панели"})
            return
        url = urlparse(self.path)
        try:
            if url.path == "/api/state":
                selected = parse_qs(url.query).get("session", [None])[0]
                result = self.server.manager.view(selected)
                result["control_token"] = self.server.token
                self._reply(200, result)
            elif url.path == "/api/alpaca/state":
                self._reply(200, self.server.manager.alpaca.view())
            elif url.path == "/api/model/state":
                self._reply(200, self.server.manager.model.view())
            elif url.path == "/api/export":
                selected = parse_qs(url.query).get("session", [None])[0]
                if not selected:
                    raise ValueError("Выберите сессию")
                view = self.server.manager.view(selected, full_events=True)
                self._reply(200, view)
            else:
                names = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "application/javascript"),
                         "/alpaca.js": ("alpaca.js", "application/javascript"),
                         "/chart.js": ("chart.js", "application/javascript"),
                         "/style.css": ("style.css", "text/css"), "/health.css": ("health.css", "text/css")}
                if url.path not in names:
                    self._reply(404, {"error": "Страница не найдена"})
                    return
                filename, mime = names[url.path]
                self._reply(200, (ASSETS / filename).read_bytes(), mime + "; charset=utf-8")
        except (ValueError, sqlite3.Error, OSError) as exc:
            self._reply(400, {"error": str(exc)})

    def do_POST(self):
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if (not self._host_allowed() or self.headers.get("X-Control-Token") != self.server.token
                or origin not in {None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"}):
            self._reply(403, {"error": "Команда разрешена только из локальной панели"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("Некорректный размер запроса")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Нужен JSON-объект")
            if self.path == "/api/model/connect":
                self._reply(200, self.server.manager.model_change(payload))
                return
            if self.path == "/api/alpaca/connect":
                result = self.server.manager.connection_change(payload=payload)
                self._reply(200, result)
                return
            elif self.path == "/api/alpaca/disconnect":
                self._reply(200, self.server.manager.connection_change())
                return
            elif self.path == "/api/start":
                session_id = self.server.manager.launch(payload=payload)
            elif self.path == "/api/sessions/delete":
                self._reply(200, self.server.manager.delete_session(payload.get("session_id")))
                return
            elif self.path in {"/api/compare", "/api/accounting/reconcile"}:
                session_id = payload.get("session_id")
                if not isinstance(session_id,str) or not session_id:
                    raise ValueError("Выберите сессию")
                kind = "comparison" if self.path == "/api/compare" else "accounting"
                self._reply(200,self.server.manager.analyze_session(session_id,kind))
                return
            elif self.path == "/api/resume":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("Выберите сессию")
                self.server.manager.launch(session_id=session_id)
            elif self.path == "/api/control":
                session_id = payload.get("session_id")
                self.server.manager.control(payload.get("action"), session_id)
            else:
                self._reply(404, {"error": "Команда не найдена"})
                return
            self._reply(200, {"session_id": session_id})
        except (ValueError, TypeError, sqlite3.Error, OSError) as exc:
            self._reply(400, {"error": str(exc)})


def serve(path: Path, port: int, *, open_browser: bool = False):
    manager = SessionManager(path)
    server = DashboardServer(("127.0.0.1", port), manager)
    print(f"Панель: http://127.0.0.1:{server.server_port} — Ctrl+C для завершения", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        server.server_close()
