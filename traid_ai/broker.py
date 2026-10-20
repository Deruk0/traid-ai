"""Durable, single-owner spot execution against Alpaca Paper only."""
from dataclasses import replace
from contextlib import nullcontext
from datetime import timedelta
from decimal import ROUND_DOWN, ROUND_UP
from uuid import uuid4
import hashlib
from pathlib import Path
import tempfile
import threading

from .alpaca import AlpacaError
from .engine import Engine
from .models import D, Quote, decimal, parse_time, stamp, utc_now
from .risk import entry_quantity
from .broker_io import BrokerReadCycle


TERMINAL = {"filled", "canceled", "expired", "rejected"}


def account_lock_path(account_id):
    identity = hashlib.sha256(("alpaca-paper:" + account_id).encode()).hexdigest()
    return Path(tempfile.gettempdir()) / "traid-ai-account-locks" / identity


class BrokerEngine(Engine):
    def __init__(self, store, session_id, client=None):
        super().__init__(store, session_id)
        self.client = client
        self.control_state = None
        self._activity_job = None
        self._strategy_deferred = False
        self.read_cycle = BrokerReadCycle(client) if client is not None else None
        if self.config.source != "alpaca-paper" or not self.state.broker:
            raise ValueError("Это не сессия исполнения Alpaca Paper")

    @classmethod
    def start_paper(cls, store, config, now, deadline, client):
        account = client.account()
        cls._active_account(account)
        if client.positions() or client.open_orders():
            raise ValueError("Для новой сессии нужен Paper-счёт без позиций и открытых заявок")
        if config.capital > min(decimal(account["cash"]),
                                decimal(account["non_marginable_buying_power"])):
            raise ValueError("Бюджет превышает доступные средства Paper-счёта")
        asset = client.asset()
        config = replace(config, quantity_step=decimal(asset["min_trade_increment"]),
                         fee_rate=max(config.fee_rate, D(".0025")),
                         # Observed Paper validation 40310000 on 2026-10-14: minimum USD 10.
                         min_notional=max(config.min_notional, D("10")))
        # Create the same compatible session record, then enrich it before any network write.
        engine = Engine.start(store, config, now, deadline)
        engine.state.broker = dict(account_id=account["account_id"], initial_cash=account["cash"],
                                   available=account["non_marginable_buying_power"],
                                   actual_cash=account["cash"], asset=asset, pending=None,
                                   last_order=None, reconciled=False, error=None,
                                   order_count=0, halted=False, last_reconcile_at=None)
        store.commit_step(engine.state, str(uuid4()), stamp(now), [("BROKER_READY", {
            "account_suffix": account["account_suffix"], "execution": "ALPACA_PAPER"})])
        return cls(store, engine.state.session_id, client)

    @staticmethod
    def _active_account(account):
        if account["status"] != "ACTIVE" or account.get("account_blocked") or account.get("trading_blocked"):
            raise ValueError("Paper-счёт не активен или торговля ограничена Alpaca")

    def _save(self, now, events):
        pending = self.state.broker.get("pending") or {}
        linked = [(kind,dict(payload,decision_id=payload.get("decision_id") or
                    pending.get("decision_id") or self.current_decision_id))
                  if kind.startswith("ORDER_") else (kind,payload) for kind,payload in events]
        self.store.commit_step(self.state, str(uuid4()), stamp(now), linked)

    def _flat(self):
        b = self.state.broker
        return D(self.state.quantity) == 0 and not b["pending"] and b["reconciled"] and not b["halted"]

    def _strategy_ready(self):
        b = self.state.broker
        ready = b["reconciled"] and not b["pending"] and not b["halted"]
        self._strategy_deferred = not ready
        return ready

    def _retry_strategy_quote(self, now):
        b = self.state.broker
        if not b["reconciled"] or b["pending"] or b["halted"]:
            return None
        if not self._strategy_deferred:
            return super()._retry_strategy_quote(now)
        self._strategy_deferred = False
        quote = self.state.quote()
        if quote is not None:
            try:
                quote.validate(now, self.config)
            except ValueError:
                return None
        return quote

    def _plan(self, side, qty, reason, now, events, *, price=None, stop_price=None, decision_id=None):
        b = self.state.broker
        if b["pending"]:
            return
        client_id = "traid-" + self.state.session_id.replace("-", "")[:24] + "-" + uuid4().hex[:12]
        request = dict(symbol="BTC/USD", side=side, qty=str(qty), client_order_id=client_id,
                       type="limit" if side == "buy" else "market",
                       time_in_force="ioc" if side == "buy" else "gtc")
        if price is not None:
            request["limit_price"] = str(price)
        b["pending"] = dict(request=request, phase="PREPARED", attempted=False, order_id=None,
                            filled_qty="0", filled_value="0", terminal=False, created_at=stamp(now),
                            position_before=self.state.quantity, cash_before=b["actual_cash"],
                            reason=reason, stop_price=str(stop_price or 0))
        b["pending"]["decision_id"] = decision_id
        if side == "buy" and self.config.strategy_version == "qwen-local-v1":
            ai = self.state.ai or {}
            if ai.get("status") == "READY" and ai.get("action") == "REQUEST_ENTRY":
                # Preserve the accepted basis across commits and PREPARED recovery.
                # A later AI status or market quote must not renew this order.
                b["pending"]["ai_basis"] = dict(snapshot_id=ai.get("snapshot_id"),
                                                observed_at=ai.get("observed_at"))
        b["reconciled"] = False
        b["order_count"] += 1
        events.append(("ORDER_PREPARED", dict(client_order_id=client_id, **{
            "side": side.upper(), "quantity": str(qty), "reason": reason,
            "decision_id": decision_id})))

    def _enter(self, quote, qty, price, stop_price, step_id, now, events):
        b = self.state.broker
        if b["pending"] or not b["reconciled"] or b["halted"] or b["order_count"] >= 100:
            events.append(("RISK_VERDICT", dict(approved=False, reason="BROKER_NOT_READY", decision_id=step_id)))
            return
        tick = decimal(b["asset"]["price_increment"])
        price = (price / tick).to_integral_value(rounding=ROUND_UP) * tick
        stop_price = price * (1 - self.config.stop_fraction)
        available = min(D(self.state.cash), decimal(b["available"]))
        allowed = entry_quantity(self.config, available, self.equity(), price, stop_price)
        maximum = (D("200000") / price / self.config.quantity_step).to_integral_value(
            rounding=ROUND_DOWN) * self.config.quantity_step
        qty = min(qty, allowed, maximum)
        if qty < decimal(b["asset"]["min_order_size"]) or qty * price < self.config.min_notional:
            events.append(("RISK_VERDICT", dict(approved=False, reason="BUDGET_OR_MINIMUM", decision_id=step_id)))
            return
        events.append(("RISK_VERDICT", dict(approved=True, quantity=str(qty),
                                            reason="WITHIN_LIMITS", limit_price=str(price), decision_id=step_id)))
        reason = "EXECUTION_TEST" if self.config.strategy_version == "paper-smoke-v1" else "STRATEGY_ENTRY"
        self._plan("buy", qty, reason, now, events, price=price, stop_price=stop_price, decision_id=step_id)

    def _exit(self, quote, reason, now, events):
        b = self.state.broker
        if reason in {"TRAILING_STOP","STOP_LOSS","MAX_HOLD"} and D(self.state.quantity) > 0:
            self.state.position_exit_reason=reason
        if b["pending"] or not b["reconciled"] or b["halted"]:
            events.append(("RISK_VERDICT", dict(approved=False, reason="BROKER_NOT_READY",
                                                decision_id=self.current_decision_id, action="REQUEST_EXIT")))
            return
        qty = D(self.state.quantity)
        if quote:
            qty = min(qty, D("200000") / quote.bid)
        qty = (qty / self.config.quantity_step).to_integral_value(rounding=ROUND_DOWN) * self.config.quantity_step
        if qty < decimal(b["asset"]["min_order_size"]):
            b["error"] = "Остаток BTC меньше минимального ордера Alpaca; закрытие не подтверждено"
            self.state.status, self.state.exit_reason = "UNRESOLVED", reason
            events.append(("BROKER_UNRESOLVED", {"reason": "RESIDUAL_BELOW_MINIMUM", "side": "sell"}))
            return
        self._plan("sell", qty, reason, now, events, decision_id=self.current_decision_id)

    def _apply_order(self, raw, now):
        p = self.state.broker["pending"]
        req = p["request"]
        if (not isinstance(raw, dict) or raw.get("client_order_id") != req["client_order_id"]
                or str(raw.get("symbol", "")).replace("/", "") != "BTCUSD"
                or raw.get("side") != req["side"] or not raw.get("id")):
            raise AlpacaError("Ответ заявки не соответствует сохранённому намерению")
        qty = decimal(raw.get("filled_qty", "0"))
        if qty < D(p["filled_qty"]) or qty > decimal(req["qty"]):
            raise AlpacaError("Количество исполнения противоречит журналу")
        average = decimal(raw.get("filled_avg_price") or "0")
        if qty > 0 and average <= 0:
            raise AlpacaError("Не подтверждена цена исполнения")
        value = qty * average
        if value < D(p["filled_value"]):
            raise AlpacaError("Сумма исполнения противоречит журналу")
        delta, delta_value = qty - D(p["filled_qty"]), value - D(p["filled_value"])
        events = []
        if delta > 0:
            p["fill_updates"] = p.get("fill_updates", 0) + 1
            estimated_fee = delta_value * self.config.fee_rate
            self.state.fees = str(D(self.state.fees) + estimated_fee)
            if req["side"] == "buy":
                if D(p["filled_qty"]) == 0:
                    self.state.entries += 1
                self.state.entered_at = self.state.entered_at or stamp(now)
                self.state.entry_price = str(average)
                new_stop=average*(1-self.config.stop_fraction)
                self.state.stop_price=str(max(D(self.state.stop_price),new_stop) if self._trailing() else new_stop)
                self.state.position_peak=str(max(D(self.state.position_peak),average))
            elif D(p["filled_qty"]) == 0:
                self.state.exits += 1
            events.append(("BROKER_FILL", dict(side=req["side"].upper(), quantity=str(delta),
                price=str(delta_value / delta), fee=str(estimated_fee), fee_is_estimated=True,
                reason=p["reason"], client_order_id=req["client_order_id"], order_id=raw["id"],
                decision_id=p.get("decision_id"))))
        p.update(order_id=str(raw["id"]), filled_qty=str(qty), filled_value=str(value),
                 phase=str(raw.get("status", "unknown")), terminal=raw.get("status") in TERMINAL)
        self.state.broker["last_order"] = dict(client_order_id=req["client_order_id"],
                                              order_id=p["order_id"], side=req["side"],
                                              status=p["phase"], filled_qty=str(qty), decision_id=p.get("decision_id"))
        events.append(("ORDER_UPDATE", dict(self.state.broker["last_order"])))
        self._save(now, events)

    def _sync(self, now, snapshot=None):
        b = self.state.broker
        b["reconciled"] = False
        account = snapshot["account"] if snapshot is not None else self.client.account()
        if account["account_id"] != b["account_id"]:
            raise AlpacaError("Подключён другой Paper-счёт; восстановление остановлено")
        positions, orders = ((snapshot["positions"], snapshot["open_orders"]) if snapshot is not None
                             else (self.client.positions(), self.client.open_orders()))
        if any(str(pos.get("symbol", "")).replace("/", "") != "BTCUSD" for pos in positions):
            b["halted"] = True
            raise AlpacaError("На выделенном Paper-счёте появилась внешняя позиция")
        p = b["pending"]
        client_id = p["request"]["client_order_id"] if p else None
        if any(order.get("client_order_id") != client_id for order in orders):
            b["halted"] = True
            raise AlpacaError("Обнаружена посторонняя открытая заявка; автоматические действия остановлены")
        q = sum((decimal(pos["qty"]) for pos in positions), D("0"))
        cash = decimal(account["cash"])
        if q < 0:
            b["halted"] = True
            raise AlpacaError("Шорт не разрешён для этой сессии")
        prior_q, prior_cash = D(self.state.quantity), D(self.state.cash)
        target_cash = self.config.capital + cash - D(b["initial_cash"])
        if p:
            before, filled, value = D(p["position_before"]), D(p["filled_qty"]), D(p["filled_value"])
            # Fee rounding can lose one asset increment per observed partial fill.
            tolerance = self.config.quantity_step * max(1, p.get("fill_updates", 1))
            if p["request"]["side"] == "buy":
                low, high = before + filled * (1 - max(self.config.fee_rate, D(".0025"))), before + filled
                cash_low = D(p["cash_before"]) - value - D(".01")
                cash_high = D(p["cash_before"]) - value + D(".01")
            else:
                low, high = max(D("0"), before - filled), max(D("0"), before - filled)
                cash_low = D(p["cash_before"]) + value * (1 - max(self.config.fee_rate, D(".0025"))) - D(".01")
                cash_high = D(p["cash_before"]) + value + D(".01")
            if q < low - tolerance or q > high + tolerance or not cash_low <= cash <= cash_high:
                detail = dict(status="MISMATCH", checked_at=stamp(now),
                              actual_quantity=str(q), expected_quantity_low=str(low-tolerance),
                              expected_quantity_high=str(high+tolerance), actual_cash=str(cash),
                              expected_cash_low=str(cash_low), expected_cash_high=str(cash_high),
                              quantity_matches=low-tolerance <= q <= high+tolerance,
                              cash_matches=cash_low <= cash <= cash_high,
                              order_terminal=p["terminal"], side=p["request"]["side"])
                b["reconciliation"] = detail
                self._save(now, [("BROKER_RECONCILIATION_MISMATCH", detail)])
                # A terminal order can remain pending until the balances match.
                # Posted fees must therefore be checked here too; otherwise the
                # pending branch permanently excludes the only reconciliation path.
                if not (p["terminal"] and not orders and self._late_fees(now,account,positions,orders)):
                    raise AlpacaError("Позиция/баланс ещё не согласованы с исполнением; ждём сверку")
                prior_q, prior_cash = D(self.state.quantity), D(self.state.cash)
        elif abs(q-prior_q) > self.config.quantity_step or abs(target_cash-prior_cash) > D(".01"):
            if not self._late_fees(now, account, positions, orders):
                b["halted"] = True
                raise AlpacaError("Баланс/остаток изменён без подтверждённого начисления; требуется сверка")
            prior_q, prior_cash = D(self.state.quantity), D(self.state.cash)
        if q > prior_q:
            self.state.entry_cost = str(D(self.state.entry_cost) + max(D("0"), prior_cash - target_cash))
        elif q < prior_q and prior_q > 0:
            cost = D(self.state.entry_cost) * (prior_q - q) / prior_q
            self.state.realized_pnl = str(D(self.state.realized_pnl) + target_cash - prior_cash - cost)
            self.state.entry_cost = str(max(D("0"), D(self.state.entry_cost) - cost))
        self.state.cash, self.state.quantity = str(target_cash), str(q)
        if q == 0:
            self.state.entry_cost = self.state.entry_price = self.state.stop_price = "0"
            self.state.entered_at = None
            self.state.position_peak,self.state.position_exit_reason="0",None
            self.state.realized_pnl = str(target_cash - self.config.capital)
        b.update(available=account["non_marginable_buying_power"], actual_cash=str(cash),
                 error=None, halted=False, reconciled=True, last_reconcile_at=stamp(now))
        if b.get("reconciliation"):
            b["reconciliation"] = dict(status="MATCHED",checked_at=stamp(now),
                                       actual_quantity=str(q),actual_cash=str(cash))
        if p and p["terminal"] and not orders:
            b["pending"] = None
        self._mark()
        self._save(now, [("BROKER_LEDGER", dict(cash=self.state.cash, quantity=self.state.quantity,
                                              fees=self.state.fees, fee_is_estimated=True))])
        return account

    def _late_fees(self, now, account, positions, orders):
        """Read-only activities explain a late fee; never invent an extra fill."""
        if not hasattr(self.client,"activities"):
            return False
        from .accounting import reconcile_activities
        start = parse_time(self.state.started_at).replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(seconds=1)
        job = self._activity_job
        if job is None or (job["done"] and (now-parse_time(job["until"])).total_seconds() >= 60):
            job = dict(done=False, rows=None, after=stamp(start), until=stamp(now))
            self._activity_job = job
            data_client = self.client.data_client() if hasattr(self.client,"data_client") else self.client
            threading.Thread(target=self._read_activities,args=(job,data_client.activities),
                             daemon=True,name="late-fee-data").start()
            return False
        if not job["done"] or job["rows"] is None:
            return False
        activities = job["rows"]
        report = reconcile_activities(self.config,self.state,self.store.events(self.state.session_id),activities,now,
                    account=account,positions=positions,open_orders=orders)
        self.state.accounting = report
        if not (report["fills_match"] and report["balances_match"] and report["has_posted_fees"]):
            self._save(now,[("ACCOUNTING_UNRESOLVED", {"reason_codes":report["reason_codes"]})])
            return False
        self.state.realized_pnl = str(D(self.state.realized_pnl)+D(report["cash"])-D(self.state.cash))
        self.state.cash, self.state.quantity = report["cash"],report["quantity"]
        self.state.fees = report["fees_usd"]
        self._save(now,[("LATE_FEES_RECONCILED",report)])
        return True

    @staticmethod
    def _read_activities(job, loader):
        try:
            job["rows"] = loader(job["after"],job["until"])
        except Exception:
            job["rows"] = None
        finally:
            job["done"] = True

    def _cancel(self, now):
        p = self.state.broker["pending"]
        if not p or p["request"]["side"] != "buy" or p["terminal"] or p.get("phase") == "CANCEL_PENDING":
            return
        if not p["attempted"]:
            self.state.broker["pending"] = None
            self._save(now, [("ORDER_ABORTED", {"reason": "ENTRY_DISABLED_BEFORE_SUBMIT"})])
        elif p["order_id"]:
            try:
                self.client.cancel_order(p["order_id"])
            except AlpacaError as exc:
                # Filled concurrently, already canceled, or cancellation still in flight.
                if exc.status not in {404, 422}:
                    raise
            p["phase"] = "CANCEL_PENDING"
            self._save(now, [("ORDER_CANCEL_REQUESTED", {"order_id": p["order_id"]})])

    def _entry_guard(self, now):
        if self.control_state:
            stop, pause = self.control_state()
            if stop or pause is True:
                self.state.status = "FLATTENING" if stop else "PAUSED_ENTRIES"
                if stop:
                    self.state.exit_reason = self.state.exit_reason or "USER_STOP"
                return "USER_STOP" if stop else "ENTRIES_PAUSED"
        actual_now = max(now, utc_now())
        if actual_now >= parse_time(self.state.deadline_at)-timedelta(seconds=self.config.flatten_buffer_seconds):
            self.state.status, self.state.exit_reason = "FLATTENING", "DEADLINE"
            return "DEADLINE"
        try:
            self.state.quote().validate(actual_now,self.config)
        except (AttributeError,ValueError):
            return "QUOTE_EXPIRED_BEFORE_SUBMIT"
        if self.config.strategy_version == "qwen-local-v1":
            basis = (self.state.broker.get("pending") or {}).get("ai_basis")
            if not isinstance(basis, dict) or not basis.get("snapshot_id"):
                return "AI_BASIS_MISSING_BEFORE_SUBMIT"
            try:
                age = (actual_now - parse_time(basis["observed_at"])).total_seconds()
            except (KeyError, AttributeError, TypeError, ValueError):
                return "AI_BASIS_MISSING_BEFORE_SUBMIT"
            if age < 0 or age > self.config.max_quote_age:
                return "AI_SNAPSHOT_EXPIRED_BEFORE_SUBMIT"
        return None

    def _dispatch(self, now):
        p = self.state.broker["pending"]
        if not p or p["attempted"]:
            return
        if p["request"]["side"] == "buy":
            reason = self._entry_guard(now)
            if reason:
                self.state.broker["pending"] = None
                self._save(now,[("ORDER_ABORTED",dict(reason=reason,decision_id=p.get("decision_id")))])
                return
        p.update(attempted=True, phase="SUBMITTING")
        self._save(now, [("ORDER_SUBMITTING", {"client_order_id": p["request"]["client_order_id"]})])
        # SQLite may stall or the computer may sleep during fsync. No POST yet:
        # recheck controls, deadline and quote before crossing the network boundary.
        # Qwen entries also retain the original AI snapshot's expiry after commit.
        if p["request"]["side"] == "buy":
            reason = self._entry_guard(now)
            if reason:
                self.state.broker["pending"] = None
                self._save(now,[("ORDER_ABORTED",dict(reason=reason,decision_id=p.get("decision_id")))])
                return
        try:
            raw = self.client.submit_order(p["request"])
        except AlpacaError as exc:
            # Definite validation/auth rejections cannot have filled. Everything else is UNKNOWN.
            if exc.status in {400, 401, 403, 422, 429}:
                self.state.broker["last_order"] = dict(client_order_id=p["request"]["client_order_id"],
                                                      status="rejected", side=p["request"]["side"], filled_qty="0")
                self.state.broker["pending"] = None
                self.state.status, self.state.exit_reason = "FLATTENING", "ORDER_REJECTED"
                self._save(now, [("ORDER_REJECTED", {"reason": str(exc)})])
                return
            raise
        self._apply_order(raw, now)

    def _confirm_absent(self, now, snapshot=None):
        """Corroborate a missing ID with account, positions, open orders and history."""
        p = self.state.broker["pending"]
        self._sync(now, snapshot)
        history = snapshot["history"] if snapshot is not None else self.client.orders()
        cid = p["request"]["client_order_id"]
        if any(order.get("client_order_id") == cid for order in history):
            raise AlpacaError("Заявка найдена в истории; ожидаем ответ по её ID")
        last = p.get("last_absence_check")
        if last is None or (now - parse_time(last)).total_seconds() >= 10:
            p["absence_checks"] = p.get("absence_checks", 0) + 1
            p["last_absence_check"] = stamp(now)
        old_enough = (now - parse_time(p["created_at"])).total_seconds() >= 60
        confirmed = old_enough and p.get("absence_checks", 0) >= 3 and D(p["filled_qty"]) == 0
        if confirmed:
            self.state.broker["last_order"] = dict(client_order_id=cid, order_id=None,
                side=p["request"]["side"], status="not_found_after_reconciliation", filled_qty="0")
            self.state.broker["pending"] = None
            self.state.status = "FLATTENING"
            self.state.exit_reason = "ORDER_NOT_ACCEPTED"
            self._save(now, [("ORDER_ABSENT_CONFIRMED", {"client_order_id":cid,
                "checks":p["absence_checks"], "reason":"NO_ORDER_OR_ACCOUNT_CHANGE"})])
        else:
            self._save(now, [("ORDER_ABSENCE_CHECK", {"client_order_id":cid,
                                                    "checks":p.get("absence_checks",0)})])
        return confirmed

    def step(self, now, quote, *, stop=False, error=None, step_id=None, pause_entries=None):
        budget = self.client.request_budget(3) if hasattr(self.client, "request_budget") else nullcontext()
        with budget:
            return self._step(now, quote, stop=stop, error=error, step_id=step_id, pause_entries=pause_entries)

    def _step(self, now, quote, *, stop=False, error=None, step_id=None, pause_entries=None):
        self.config, self.state = self.store.load(self.state.session_id)
        if self.state.status == "CLOSED":
            return []
        b = self.state.broker
        try:
            p = b["pending"]
            client_id = p["request"]["client_order_id"] if p and p["attempted"] else None
            snapshot = self.read_cycle.poll(client_id)
            b["io"] = self.read_cycle.view()
            if snapshot is None:
                b["reconciled"] = False
                self._save(now, [])
                # Controls, fresh quotes and protective checks still run, with entry/exit
                # submission gated by reconciliation. UNKNOWN cannot be bypassed.
                return super().step(now, quote, stop=stop, error=error, step_id=step_id,
                                    pause_entries=pause_entries)
            if p and p["attempted"]:
                raw = snapshot["order"]
                if raw is None:
                    if not self._confirm_absent(now, snapshot):
                        raise AlpacaError("Результат отправки не подтверждён; выполняется сверка, повторная покупка запрещена")
                else:
                    self._apply_order(raw, now)
            closing = stop or self.state.status in {"FLATTENING", "UNRESOLVED"} or now >= (
                parse_time(self.state.deadline_at) - timedelta(seconds=self.config.flatten_buffer_seconds))
            if closing or pause_entries is True or self.state.position_exit_reason:
                self._cancel(now)
            account = self._sync(now, snapshot)
            if account.get("trading_blocked") or account.get("account_blocked"):
                raise AlpacaError("Alpaca ограничила торговлю; требуется восстановление")
        except (AlpacaError, ValueError) as exc:
            b = self.state.broker
            b["io"] = self.read_cycle.view()
            b.update(error=str(exc), reconciled=False)
            self.state.status = "UNRESOLVED"
            self.state.exit_reason = self.state.exit_reason or ("USER_STOP" if stop else "BROKER_RECONCILIATION")
            self.state.last_tick_at = stamp(now)
            self._save(now, [("BROKER_UNRESOLVED", {"reason": str(exc)})])
            return []
        if (self.config.strategy_version == "paper-smoke-v1" and self.state.entries > 0
                and self._flat()):
            stop = True
            self.state.exit_reason = self.state.exit_reason or "EXECUTION_TEST_COMPLETE"
            self._save(now, [])
        events = super().step(now, quote, stop=stop, error=error,
                              step_id=step_id or "broker-tick:" + stamp(now),
                              pause_entries=pause_entries)
        # A reduction-only market exit does not invent a fill from an old market quote.
        if (self.state.status in {"FLATTENING", "UNRESOLVED"} or self.state.position_exit_reason) and D(self.state.quantity) > 0:
            extra = []
            self._exit(self.state.quote(), self.state.exit_reason or self.state.position_exit_reason or "RECOVERY", now, extra)
            if extra:
                self._save(now, extra)
        try:
            if self.state.position_exit_reason:
                self._cancel(now)
            self._dispatch(now)
        except (AlpacaError, ValueError) as exc:
            p = self.state.broker["pending"]
            if p:
                p["phase"] = "UNKNOWN"
            self.state.broker.update(error=str(exc), reconciled=False)
            self.state.status, self.state.exit_reason = "UNRESOLVED", self.state.exit_reason or "BROKER_RECONCILIATION"
            self._save(now, [("ORDER_UNKNOWN", {"reason": str(exc)})])
        return events

    def close(self):
        if self.read_cycle:
            self.read_cycle.close()

    def report(self):
        result = super().report()
        result.update(mode="ALPACA_PAPER", execution_model="Alpaca Paper acknowledged orders and reconciled positions",
                      fees_are_estimated=True, pnl_is_final=False,
                      fee_note="Комиссии оценены; поздние начисления Alpaca требуют отдельной сверки")
        if self.state.accounting:
            result.update(accounting=self.state.accounting,
                          pnl_is_final=self.state.accounting.get("pnl_is_final",False),
                          fees_are_estimated=self.state.accounting.get("fees_are_estimated",True),
                          fee_note="Сверено по начислениям на указанное время" if self.state.accounting.get("pnl_is_final") else
                          "Начисления проверены, но окончательная сверка не подтверждена")
        return result
