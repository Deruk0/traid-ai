from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from .models import Config, D, Quote, State, parse_time, stamp
from .risk import entry_quantity
from .store import Store
from .strategy import DemoMomentum, Intent, ExecutionSmoke
from .analytics import remember_quote
from .objectives import session_objective


class Engine:
    def __init__(self, store: Store, session_id: str):
        self.store = store
        self.config, self.state = store.load(session_id)
        self.candle_provider = None
        self.current_decision_id = None
        if self.config.strategy_version == "qwen-local-v1":
            from .qwen import QwenStrategy
            self.strategy = QwenStrategy(model=self.config.llm_model)
        else:
            self.strategy = ExecutionSmoke(lambda: self.state) if self.config.strategy_version == "paper-smoke-v1" else DemoMomentum()

    @classmethod
    def start(cls, store: Store, config: Config, now: datetime, deadline: datetime):
        if config.strategy_version == "qwen-local-v1" and not config.llm_digest:
            raise ValueError("Сначала подключи и проверь Qwen в панели")
        if (deadline - now).total_seconds() <= config.flatten_buffer_seconds:
            raise ValueError("Сессия должна быть длиннее буфера завершения")
        state = State(str(uuid4()), stamp(now), stamp(deadline), "RUNNING",
                      str(config.capital), str(config.capital))
        store.create(config, state)
        return cls(store, state.session_id)

    def equity(self) -> Decimal:
        quote = self.state.quote()
        return D(self.state.cash) + D(self.state.quantity) * (quote.bid if quote else D("0"))

    def _mark(self):
        equity = self.equity()
        peak = max(equity, D(self.state.peak_equity))
        self.state.peak_equity = str(peak)
        self.state.max_drawdown = str(max(D(self.state.max_drawdown), (peak - equity) / peak))

    def _trailing(self):
        return self.config.exit_policy == "trailing-stop-v1"

    def _update_trailing(self, quote, events):
        if not self._trailing() or D(self.state.quantity) <= 0:
            return
        peak=max(D(self.state.position_peak),D(self.state.entry_price),quote.bid)
        previous=D(self.state.stop_price)
        stop=max(previous,peak*(1-self.config.stop_fraction))
        self.state.position_peak,self.state.stop_price=str(peak),str(stop)
        if stop > previous:
            events.append(("TRAILING_STOP_MOVED",dict(previous_stop=str(previous),stop_price=str(stop),
                position_peak=str(peak),quote_id=quote.quote_id)))

    def _protective_reason(self, quote, now):
        if D(self.state.quantity) <= 0:
            return None
        if self.state.position_exit_reason:
            return self.state.position_exit_reason
        if quote is not None and quote.bid <= D(self.state.stop_price):
            return "TRAILING_STOP" if self._trailing() else "STOP_LOSS"
        if (self.config.max_hold_seconds is not None and self.state.entered_at
                and (now-parse_time(self.state.entered_at)).total_seconds() >= self.config.max_hold_seconds):
            return "MAX_HOLD"
        return None

    def _exit(self, quote: Quote, reason: str, now: datetime, events: list):
        qty = D(self.state.quantity)
        if qty == 0:
            return
        price = quote.bid * (1 - self.config.slippage_bps / 10000)
        fee = qty * price * self.config.fee_rate
        proceeds = qty * price - fee
        pnl = proceeds - D(self.state.entry_cost)
        self.state.cash = str(D(self.state.cash) + proceeds)
        self.state.realized_pnl = str(D(self.state.realized_pnl) + pnl)
        self.state.fees = str(D(self.state.fees) + fee)
        self.state.slippage_cost = str(D(self.state.slippage_cost) + qty * (quote.bid - price))
        self.state.quantity = self.state.entry_cost = self.state.entry_price = self.state.stop_price = "0"
        self.state.entered_at = None
        self.state.position_peak,self.state.position_exit_reason="0",None
        self.state.exits += 1
        events.append(("PAPER_FILL", dict(side="SELL", quantity=str(qty), price=str(price),
                                        fee=str(fee), realized_pnl=str(pnl), reason=reason,
                                        quote_id=quote.quote_id)))
        self._mark()

    def _flat(self):
        return D(self.state.quantity) == 0

    def _strategy_ready(self):
        return True

    def _retry_strategy_quote(self, now):
        if not hasattr(self.strategy, "has_ready_result") or not self.strategy.has_ready_result():
            return None
        quote = self.state.quote()
        if quote is not None:
            try:
                quote.validate(now, self.config)
            except ValueError:
                return None
        return quote

    def _enter(self, quote, qty, price, stop_price, step_id, now, events):
        fee = qty * price * self.config.fee_rate
        cost = qty * price + fee
        self.state.cash = str(D(self.state.cash) - cost)
        self.state.quantity, self.state.entry_price = str(qty), str(price)
        self.state.position_peak,self.state.position_exit_reason=str(price),None
        self.state.entry_cost, self.state.stop_price = str(cost), str(stop_price)
        self.state.entered_at = stamp(now)
        self.state.fees = str(D(self.state.fees) + fee)
        self.state.slippage_cost = str(D(self.state.slippage_cost) + qty * (price - quote.ask))
        self.state.entries += 1
        events.append(("PAPER_FILL", {"side": "BUY", "quantity": str(qty),
                                     "price": str(price), "fee": str(fee),
                                     "quote_id": quote.quote_id, "decision_id": step_id}))
        self._mark()

    def step(self, now: datetime, quote: Quote | None, *, stop: bool = False,
             error: str | None = None, step_id: str | None = None,
             pause_entries: bool | None = None) -> list[tuple[str, dict]]:
        step_id = step_id or (quote.quote_id if quote else str(uuid4()))
        self.current_decision_id = step_id
        if self.store.seen(self.state.session_id, step_id):
            return []
        # Reload durable state before every transaction; failed commits don't leak local state.
        self.config, self.state = self.store.load(self.state.session_id)
        if self.state.status == "CLOSED":
            return []
        if now < parse_time(self.state.started_at):
            raise ValueError("Время раньше начала сессии")
        if self.state.last_tick_at and now < parse_time(self.state.last_tick_at):
            raise ValueError("Время обработки не может идти назад")
        events = []
        previous_status = self.state.status
        if pause_entries is not None and self.state.status in {"RUNNING", "PAUSED_ENTRIES"}:
            self.state.status = "PAUSED_ENTRIES" if pause_entries else "RUNNING"
        deadline = parse_time(self.state.deadline_at)
        flatten_at = deadline - timedelta(seconds=self.config.flatten_buffer_seconds)
        self.state.last_tick_at = stamp(now)
        valid = False
        if quote:
            try:
                quote.validate(now, self.config)
                previous = self.state.quote()
                if previous and parse_time(quote.observed_at) <= parse_time(previous.observed_at):
                    raise ValueError("Котировка повторяет или предшествует предыдущей")
                if previous and (now - parse_time(previous.observed_at)).total_seconds() > self.config.max_quote_age:
                    self.state.history = []
                self.state.last_quote = {k: str(v) for k, v in asdict(quote).items()}
                if self.config.strategy_version == "qwen-local-v1":
                    remember_quote(self.state, quote, now)
                    if self.candle_provider:
                        context = self.candle_provider(now)
                        if context.get("rows"):
                            prior = self.state.candle_context.get("summary", {}).get("provenance", {})
                            current = context["summary"]["provenance"]
                            self.state.candle_context = context
                            if prior.get("candle_set_sha256") != current.get("candle_set_sha256"):
                                events.append(("CANDLE_HISTORY", context))
                events.append(("SNAPSHOT", asdict(quote)))
                valid = True
                self._mark()
                self._update_trailing(quote,events)
            except ValueError as exc:
                events.append(("DATA_REJECTED", {"reason": str(exc), "quote_id": quote.quote_id}))
        if error:
            events.append(("DATA_UNAVAILABLE", {"reason": error}))
        new_quote = valid
        if quote is None:
            # A broker may finish reconciliation between market updates. Retry only
            # its deferred decision; this is not a new observation or history sample.
            quote = self._retry_strategy_quote(now)
            valid = quote is not None
            if valid:
                events.append(("QUOTE_RECHECK", dict(quote_id=quote.quote_id,
                                                      observed_at=quote.observed_at)))
        if stop or now >= flatten_at:
            self.state.exit_reason = self.state.exit_reason or ("USER_STOP" if stop else "DEADLINE")
            self.state.status = "FLATTENING"
        if valid:
            equity = self.equity()
            drawdown = (D(self.state.peak_equity) - equity) / D(self.state.peak_equity)
            if (self.config.capital - equity) / self.config.capital >= self.config.loss_limit:
                self.state.status, self.state.exit_reason = "FLATTENING", "SESSION_LOSS_LIMIT"
            elif drawdown >= self.config.drawdown_limit:
                self.state.status, self.state.exit_reason = "FLATTENING", "DRAWDOWN_LIMIT"
        # Time protection does not require a market price. Local fills still do;
        # a broker can reduce a reconciled position without inventing that price.
        reason = self._protective_reason(quote if valid else None, now)
        if reason and not valid and not self.state.position_exit_reason:
            self.state.position_exit_reason = reason
            events.append(("PROTECTIVE_EXIT_REQUIRED", dict(reason=reason, decision_id=step_id)))
        if self.state.status in {"FLATTENING", "UNRESOLVED"}:
            if D(self.state.quantity) > 0 and valid:
                events.append(("DECISION", dict(action="REQUEST_EXIT", reason=self.state.exit_reason or "RECOVERY",
                    decision_id=step_id, quote_id=quote.quote_id, strategy_version="independent-control")))
                events.append(("RISK_VERDICT", dict(approved=True, quantity=self.state.quantity,
                    reason="REDUCE_ONLY", decision_id=step_id)))
                self._exit(quote, self.state.exit_reason or "RECOVERY", now, events)
            if self._flat():
                self.state.status = "CLOSED"
            elif now >= deadline or stop:
                self.state.status = "UNRESOLVED"
        elif valid:
            has_position = D(self.state.quantity) > 0
            if reason:
                intent = Intent("REQUEST_EXIT", reason)
            elif has_position and self._trailing() and self.config.strategy_version != "paper-smoke-v1":
                intent = Intent("WAIT","TRAILING_POSITION_HOLD")
            elif not self._strategy_ready():
                intent = Intent("WAIT", "BROKER_RECONCILING")
            else:
                try:
                    if hasattr(self.strategy, "decide_snapshot"):
                        intent, ai_events = self.strategy.decide_snapshot(self.config, self.state, quote, now)
                        events.extend(ai_events)
                    else:
                        intent = self.strategy.decide(self.state.history, quote.mid, has_position)
                except Exception:
                    # Decision failures never prevent subsequent deadline/stop handling.
                    intent = Intent("WAIT", "DECISION_UNAVAILABLE")
                    events.append(("DECISION_UNAVAILABLE", {"quote_id": quote.quote_id}))
            if intent.action == "REQUEST_ENTRY" and self.state.status == "PAUSED_ENTRIES":
                intent = Intent("WAIT", "ENTRIES_PAUSED")
            events.append(("DECISION", {**asdict(intent), "decision_id": step_id,
                                         "ai_snapshot_id": self.state.ai.get("snapshot_id") if self.state.ai and intent.reason == "QWEN_DECISION" else None,
                                         "quote_id": quote.quote_id,
                                         "strategy_version": "independent-control" if reason or intent.reason == "TRAILING_POSITION_HOLD" else self.strategy.version}))
            if reason or intent.action == "REQUEST_EXIT":
                events.append(("RISK_VERDICT", dict(approved=True, quantity=self.state.quantity,
                    reason="REDUCE_ONLY", decision_id=step_id)))
                self._exit(quote, reason or intent.reason, now, events)
            elif intent.action == "REQUEST_ENTRY" and not has_position and self.state.status == "RUNNING":
                price = quote.ask * (1 + self.config.slippage_bps / 10000)
                stop_price = price * (1 - self.config.stop_fraction)
                qty = entry_quantity(self.config, D(self.state.cash), self.equity(), price, stop_price)
                events.append(("RISK_VERDICT", {"approved": qty > 0, "quantity": str(qty),
                                                "reason": "WITHIN_LIMITS" if qty else "BUDGET_OR_MINIMUM",
                                                "decision_id": step_id}))
                if qty > 0:
                    self._enter(quote, qty, price, stop_price, step_id, now, events)
        if new_quote:
            self.state.history = (self.state.history + [str(quote.mid)])[-5:]
        if hasattr(self.strategy, "invalidate") and (reason or self.state.status != "RUNNING"
                                                     or pause_entries is True):
            self.strategy.invalidate()
            if self.state.ai and self.state.ai.get("status") == "PENDING":
                self.state.ai = dict(self.state.ai, status="DISCARDED", reason="Приоритет управления риском")
        if previous_status != self.state.status:
            events.append(("SESSION_STATUS", {"from": previous_status, "to": self.state.status,
                                               "reason": self.state.exit_reason}))
        self.store.commit_step(self.state, step_id, stamp(now), events)
        return events

    def report(self) -> dict:
        equity = self.equity()
        mark_quote = self.state.quote()
        return {"session_id": self.state.session_id, "mode": "PAPER_ONLY",
                "source": self.config.source, "instrument": "BTC/USD", "currency": "USD",
                "status": self.state.status, "deadline_at": self.state.deadline_at,
                "capital": str(self.config.capital), "cash": self.state.cash,
                "quantity": self.state.quantity, "equity_at_last_quote": str(equity),
                "valuation_observed_at": mark_quote.observed_at if mark_quote else None,
                "net_pnl_at_last_quote": str(equity - self.config.capital),
                "realized_pnl": self.state.realized_pnl,
                "unrealized_pnl_at_last_quote": str(equity - self.config.capital - D(self.state.realized_pnl)),
                "fees": self.state.fees, "modeled_slippage_cost": self.state.slippage_cost,
                "max_drawdown": self.state.max_drawdown, "entries": self.state.entries,
                "exits": self.state.exits, "exit_reason": self.state.exit_reason,
                "strategy": self.config.strategy_version,
                "exit_policy":self.config.exit_policy,"position_peak":self.state.position_peak,
                "trailing_percent":str(self.config.stop_fraction*100),
                "objective":session_objective(self.config,self.state),
                "execution_model": "immediate full virtual fills; not an exchange backtest",
                "broker": self.state.broker, "ai": self.state.ai, "accounting": self.state.accounting}
