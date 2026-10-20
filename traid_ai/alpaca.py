"""Alpaca Paper adapter. Credentials stay in process memory."""
from copy import deepcopy
from contextlib import contextmanager
import json
import os
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, HTTPRedirectHandler, build_opener
from urllib.parse import quote as urlquote, urlencode
from datetime import timedelta

from .models import decimal, parse_time, stamp, utc_now


PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward authentication headers to a redirected destination.
        return None


class AlpacaError(ValueError):
    def __init__(self, message, *, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class PaperClient:
    def __init__(self, key: str, secret: str):
        for value in (key, secret):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", value):
                raise AlpacaError("Нужны Key ID и Secret Key из Alpaca Paper Only")
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                         "Accept": "application/json", "User-Agent": "TraidAI-Paper/0.4"}
        self._opener = build_opener(NoRedirects())
        self._io_lock = threading.RLock()
        self._budget = threading.local()

    @contextmanager
    def request_budget(self, seconds):
        previous = getattr(self._budget, "deadline", None)
        deadline = time.monotonic() + seconds
        self._budget.deadline = min(previous, deadline) if previous is not None else deadline
        try:
            yield
        finally:
            self._budget.deadline = previous

    def _timeout(self):
        deadline = getattr(self._budget, "deadline", None)
        remaining = deadline-time.monotonic() if deadline is not None else 10
        if remaining <= 0:
            raise AlpacaError("Истёк бюджет запроса Alpaca")
        return min(10, remaining)

    def _get(self, path: str, *, market_data: bool = False):
        try:
            return self._request("GET", path, market_data=market_data)

        except AlpacaError as exc:
            if not exc.retryable or getattr(self._budget, "deadline", None) is not None:
                raise
            # One bounded retry for reads only; a lost POST response remains UNKNOWN.
            return self._request("GET", path, market_data=market_data)

    def data_client(self):
        # Separate I/O lock: data loading must not hold the executor's lock.
        return PaperClient(self._headers["APCA-API-KEY-ID"], self._headers["APCA-API-SECRET-KEY"])

    def _request(self, method, path, *, market_data=False, payload=None):
        # Fixed Paper/data hosts; redirects forbidden, no live endpoint.
        base = DATA_URL if market_data else PAPER_URL
        headers = dict(self._headers)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(base + path, headers=headers, method=method, data=data)
        try:
            if not self._io_lock.acquire(timeout=self._timeout()):
                raise AlpacaError("Канал Alpaca занят; запрос не отправлен")
            try:
                with self._opener.open(request, timeout=self._timeout()) as response:
                    body = response.read(1024 * 1024 + 1)
            finally:
                self._io_lock.release()
            if not body and method == "DELETE":
                return None
            if len(body) > 1024 * 1024:
                raise AlpacaError("Ответ Alpaca слишком большой")
            return json.loads(body)
        except HTTPError as exc:
            messages = {401: "Alpaca не приняла ключи. Проверь ключи именно Paper Only",
                        403: "Alpaca отклонила запрос: HTTP 403",
                        429: "Лимит запросов Alpaca. Подождите следующего обновления"}
            # Only the documented message/code, with both credentials redacted before truncation.
            detail = ""
            try:
                error_body = json.loads(exc.read(4096))
                if isinstance(error_body, dict) and isinstance(error_body.get("message"), str):
                    message = error_body["message"]
                    for name in ("APCA-API-KEY-ID", "APCA-API-SECRET-KEY"):
                        message = message.replace(self._headers[name], "[redacted]")
                    message = " ".join(message.split())[:240]
                    code = str(error_body.get("code", ""))
                    detail = " — " + (code + ": " if code.isdigit() else "") + message
            except (OSError, ValueError, UnicodeError):
                pass
            finally:
                exc.close()
            raise AlpacaError(messages.get(exc.code, f"Alpaca недоступна: HTTP {exc.code}") + detail,
                              status=exc.code) from None
        except (URLError, TimeoutError, OSError):
            raise AlpacaError("Нет связи с Alpaca. Проверь интернет и повтори подключение",
                              retryable=True) from None
        except (json.JSONDecodeError, UnicodeError):
            raise AlpacaError("Alpaca вернула некорректный ответ") from None

    def account(self):
        raw = self._get("/v2/account")
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw.get("status"):
            raise AlpacaError("Не удалось подтвердить Paper-счёт")
        try:
            amounts = {name: str(decimal(raw[name])) for name in ("equity", "cash", "buying_power")}
        except (KeyError, ValueError):
            raise AlpacaError("Неполные данные баланса Alpaca") from None
        return dict(**amounts, account_id=raw["id"], account_suffix=raw["id"][-6:], status=str(raw["status"]),
                    crypto_status=str(raw.get("crypto_status", "UNKNOWN")),
                    non_marginable_buying_power=str(decimal(raw.get("non_marginable_buying_power", raw["cash"]))),
                    currency=str(raw.get("currency", "USD")),
                    trading_blocked=bool(raw.get("trading_blocked", False)),
                    account_blocked=bool(raw.get("account_blocked", False)))

    def positions(self):
        raw = self._get("/v2/positions")
        return self._rows(raw, ("symbol", "asset_class", "qty", "avg_entry_price", "market_value",
                                "unrealized_pl", "side"), 100)

    def orders(self):
        raw = self._get("/v2/orders?status=all&limit=50&direction=desc")
        return self._rows(raw, ("id", "client_order_id", "symbol", "side", "type", "qty", "notional", "filled_qty",
                                "filled_avg_price", "status", "submitted_at", "filled_at"), 50)

    def open_orders(self):
        return self._rows(self._get("/v2/orders?status=open&limit=500"),
                          ("id", "client_order_id", "symbol", "side", "status"), 500)

    def asset(self):
        raw = self._get("/v2/assets/BTCUSD")
        try:
            if raw["class"] != "crypto" or raw["status"] != "active" or not raw["tradable"]:
                raise ValueError()
            values = {key: str(decimal(raw[key])) for key in
                      ("min_order_size", "min_trade_increment", "price_increment")}
            if any(decimal(v) <= 0 for v in values.values()):
                raise ValueError()
            return values
        except (KeyError, TypeError, ValueError):
            raise AlpacaError("BTC/USD сейчас недоступен для Paper-торговли") from None

    def order_by_client_id(self, client_order_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", client_order_id):
            raise AlpacaError("Некорректный идентификатор заявки")
        try:
            return self._get("/v2/orders:by_client_order_id?client_order_id=" + urlquote(client_order_id))
        except AlpacaError as exc:
            if exc.status == 404:
                return None
            raise

    def submit_order(self, order):
        # Narrow contract: one spot instrument, one immutable client ID, no live host.
        if (order.get("symbol") != "BTC/USD" or order.get("side") not in {"buy", "sell"}
                or order.get("type") not in {"limit", "market"}
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", order.get("client_order_id", ""))
                or decimal(order.get("qty")) <= 0):
            raise AlpacaError("Некорректная Paper-заявка")
        if order["side"] == "buy" and (order["type"] != "limit" or order.get("time_in_force") != "ioc"):
            raise AlpacaError("Вход допускается только лимитной IOC-заявкой")
        if order["type"] == "limit" and decimal(order.get("limit_price")) <= 0:
            raise AlpacaError("Нужна положительная лимитная цена")
        allowed = {"symbol", "side", "type", "qty", "time_in_force", "limit_price", "client_order_id"}
        if set(order) - allowed:
            raise AlpacaError("Неподдерживаемые параметры заявки")
        return self._request("POST", "/v2/orders", payload=order)

    def cancel_order(self, order_id):
        if not isinstance(order_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", order_id):
            raise AlpacaError("Некорректный ID заявки")
        return self._request("DELETE", "/v2/orders/" + urlquote(order_id))

    @staticmethod
    def _rows(raw, fields, limit):
        if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
            raise AlpacaError("Некорректный список от Alpaca")
        return [{key: row.get(key) for key in fields if row.get(key) is None or
                 isinstance(row.get(key), (str, int, float, bool))} for row in raw[:limit]]

    def quote(self):
        raw = self._get("/v1beta3/crypto/us/latest/quotes?symbols=BTC%2FUSD", market_data=True)
        try:
            q = raw["quotes"]["BTC/USD"]
            bid, ask = decimal(q["bp"]), decimal(q["ap"])
            when = parse_time(q["t"])
            if bid <= 0 or ask < bid:
                raise ValueError()
            age = (utc_now() - when).total_seconds()
            return dict(symbol="BTC/USD", bid=str(bid), ask=str(ask), observed_at=stamp(when),
                        stale=age < -5 or age > 30)
        except (KeyError, TypeError, ValueError):
            raise AlpacaError("Некорректная котировка BTC/USD от Alpaca") from None

    def candles(self, now):
        from .history import parse_alpaca_candles
        query = urlencode(dict(symbols="BTC/USD", timeframe="1Min", limit=300,
                               start=stamp(now-timedelta(hours=4)), end=stamp(now), sort="asc"))
        raw = self._get("/v1beta3/crypto/us/bars?" + query, market_data=True)
        if not isinstance(raw, dict) or raw.get("next_page_token"):
            raise AlpacaError("Свечная история Alpaca неполна")
        return parse_alpaca_candles(raw, utc_now())

    def activities(self, after, until):
        """Read a complete bounded activity range; never silently truncate pages."""
        start, end = parse_time(after), parse_time(until)
        if start > end:
            raise AlpacaError("Некорректный период начислений")
        rows, token, tokens = [], None, set()
        fields = ("id", "activity_type", "order_id", "symbol", "side", "qty", "price",
                  "net_amount", "transaction_time", "date", "status", "currency")
        for _ in range(20):
            params = dict(after=stamp(start), until=stamp(end), direction="asc", page_size=100)
            if token:
                params["page_token"] = token
            raw = self._get("/v2/account/activities?" + urlencode(params))
            if not isinstance(raw, list) or len(raw) > 100 or any(not isinstance(r, dict) for r in raw):
                raise AlpacaError("Некорректные начисления Alpaca")
            page = self._rows(raw, fields, 100)
            if any(not isinstance(r.get("id"), str) or not r.get("activity_type") for r in page):
                raise AlpacaError("Начисление не имеет идентификатора")
            rows.extend(page)
            if len(page) < 100:
                return rows
            token = page[-1]["id"]
            if token in tokens:
                raise AlpacaError("Повтор страницы начислений Alpaca")
            tokens.add(token)
        raise AlpacaError("Начислений слишком много; полная сверка не подтверждена")


class PaperConnection:
    def __init__(self, *, client_factory=PaperClient, poll_seconds=15):
        self._factory = client_factory
        self._poll_seconds = poll_seconds
        self._lock = threading.RLock()
        self._connect_lock = threading.Lock()
        self._generation = 0
        self._wake = threading.Event()
        self._thread = None
        self._state = self._empty()
        self._client = None

    @staticmethod
    def _empty():
        return dict(connected=False, mode="paper", endpoint=PAPER_URL,
                    account=None, positions=[], orders=[], quote=None, updated_at=None,
                    refreshing=False, errors={}, resource_times={}, environment_keys_available=bool(
                        os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")))

    def connect(self, key=None, secret=None):
        with self._connect_lock:
            if key is None and secret is None:
                key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
            client = self._factory(key, secret)
            account = client.account()  # Confirm against paper endpoint before claiming connected.
            with self._lock:
                self._generation += 1
                generation = self._generation
                self._wake.set()
                wake = threading.Event()
                self._wake = wake
                self._state = dict(self._empty(), connected=True, account=account, refreshing=True)
                self._client = client
                self._state["resource_times"]["account"] = stamp(utc_now())
                self._thread = threading.Thread(target=self._poll, args=(client, generation, wake),
                                                name="alpaca-paper-reader", daemon=True)
                self._thread.start()
            return self.view()

    def _poll(self, client, generation, wake):
        while not wake.is_set():
            result, errors = {}, {}
            for name in ("account", "positions", "orders", "quote"):
                if wake.is_set():
                    return
                try:
                    result[name] = getattr(client, name)()
                except ValueError as exc:
                    errors[name] = str(exc)
                except Exception:
                    errors[name] = "Не удалось обработать ответ Alpaca"
            with self._lock:
                if generation != self._generation:
                    return
                self._state.update(result, errors=errors, refreshing=False,
                                   updated_at=stamp(utc_now()))
                for name in result:
                    self._state["resource_times"][name] = stamp(utc_now())
                # Failure leaves last known data visible with explicit stale markers.
            wake.wait(self._poll_seconds)

    def view(self):
        with self._lock:
            state = deepcopy(self._state)
            if state["quote"]:
                age = (utc_now() - parse_time(state["quote"]["observed_at"])).total_seconds()
                state["quote"]["stale"] = age < -5 or age > 30
            return state

    def disconnect(self):
        with self._connect_lock:
            with self._lock:
                self._generation += 1
                self._wake.set()
                self._state = self._empty()
                self._client = None
        # Does not cancel orders or close any remote positions.

    def shutdown(self):
        self.disconnect()
        if self._thread:
            self._thread.join(timeout=6)

    def execution_client(self):
        with self._lock:
            if not self._client or not self._state["connected"]:
                raise AlpacaError("Сначала подключите Alpaca Paper в панели")
            return self._client
