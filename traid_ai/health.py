"""Read-only technical diagnostics. These observations do not authorize orders."""
from .models import parse_time, stamp


def technical_health(config, state, events, now, *, active=False, watchdog=None,
                     quote_io=None, broker_io=None):
    watchdog = watchdog or {}
    archived = state is not None and not active
    recovery = archived and state.status != "CLOSED"
    reference = parse_time(state.last_tick_at or state.started_at) if archived else now
    rows, blockers = [], []

    def row(key, label, status, detail, **values):
        rows.append(dict(component=key, label=label, status=status, detail=detail, **values))

    def block(code, text):
        blockers.append(dict(code=code, text=text))

    if not state:
        return dict(scope="IDLE", as_of=stamp(now), components=[], blockers=[], latest_decision=None)
    if not active:
        block("NOT_RUNNING", "Выбранная сессия сейчас не выполняется")
    elif state.status != "RUNNING":
        block("SESSION_CONTROL", "Пауза или завершение сессии запрещает новые входы")
    age = max(0, (reference-parse_time(state.last_tick_at)).total_seconds()) if state.last_tick_at else None
    worker_status = "RECOVERY_REQUIRED" if recovery else "ARCHIVE" if archived else "ERROR" if watchdog.get("tripped") else "READY"
    row("worker", "Исполнитель", worker_status,
        "Сессия не завершена; в этой панели исполнитель не запущен" if recovery else
        "Сохранённое состояние сессии" if archived else watchdog.get("reason") or "Владелец исполнения работает",
        last_progress_at=state.last_tick_at, age_seconds=age)
    if active and watchdog.get("tripped"):
        block("WATCHDOG", "Watchdog обнаружил остановку прогресса или пробуждение компьютера")
    quote = state.quote()
    quote_age = (reference-parse_time(quote.observed_at)).total_seconds() if quote else None
    quote_status = ("SIMULATED" if config.source == "synthetic" else "UNAVAILABLE" if quote is None else
                    "STALE" if quote_age > config.max_quote_age or quote_age < -5 else "READY")
    row("quotes", "Котировки", quote_status,
        "Искусственные цены" if config.source == "synthetic" else
        "Нет принятой котировки" if quote is None else "Возраст последней принятой котировки",
        observed_at=quote.observed_at if quote else None, age_seconds=quote_age,
        max_age_seconds=config.max_quote_age, io=quote_io if active else None)
    if active and quote_status in {"UNAVAILABLE", "STALE"}:
        block("QUOTE_"+quote_status, "Для входа нужна свежая проверенная котировка")
    candles = state.candle_context.get("summary", {})
    candle_age = ((reference-parse_time(candles["latest_end_at"])).total_seconds()
                  if candles.get("latest_end_at") else None)
    candle_status = "NOT_USED" if config.strategy_version != "qwen-local-v1" else (
        "STALE" if candle_age is not None and candle_age > 180 else candles.get("status", "UNAVAILABLE"))
    row("candles", "Свечная история", candle_status, "Готовность закрытых свечей",
        count=candles.get("count", 0), age_seconds=candle_age)
    if config.source == "alpaca-paper":
        b = state.broker
        io = broker_io if active and broker_io is not None else b.get("io", {})
        status = ("ERROR" if b.get("halted") else "TIMED_OUT" if io.get("status") == "TIMED_OUT" else
                  "READING" if io.get("status") == "READING" else "ERROR" if b.get("error") else
                  "READY" if b.get("reconciled") else "UNCONFIRMED")
        reconcile_age = ((reference-parse_time(b["last_reconcile_at"])).total_seconds()
                         if b.get("last_reconcile_at") else None)
        row("broker", "Сверка брокера", status,
            b.get("error") or "Баланс, позиции и ордера сверяются вместе",
            last_success_at=b.get("last_reconcile_at"), age_seconds=reconcile_age, io=io,
            pending_phase=(b.get("pending") or {}).get("phase"),reconciliation=b.get("reconciliation"))
        if active and (not b.get("reconciled") or status != "READY"):
            block("BROKER_UNCONFIRMED", "Ожидается подтверждённая сверка брокера")
        if active and b.get("pending"):
            block("ORDER_PENDING", "Предыдущая заявка ещё не завершена или требует сверки")
    ai = state.ai or {}
    last_answer = next((e["payload"] for e in reversed(events) if e["kind"] == "AI_DECISION"), {})
    ai_age = ((reference-parse_time(ai["processed_at"])).total_seconds() if ai.get("processed_at") else None)
    ai_status = "NOT_USED" if config.strategy_version != "qwen-local-v1" else ai.get("status", "WARMUP")
    if recovery and ai_status == "PENDING":
        ai_status = "INTERRUPTED"
    row("qwen", "Qwen", ai_status, "Состояние последнего запроса модели",
        waiting_seconds=max(0, ai_age) if ai_age is not None and ai_status == "PENDING" else None,
        latency_ms=ai.get("latency_ms") if ai.get("latency_ms") is not None else last_answer.get("latency_ms"),
        error_code=ai.get("error_code"), elapsed_ms=ai.get("elapsed_ms"),
        result_wait_ms=ai.get("result_wait_ms"), decision_age_ms=ai.get("decision_age_ms"),
        processed_at=ai.get("processed_at"), review_at=ai.get("review_at"),
        review_trigger=ai.get("review_trigger"), wait_streak=ai.get("wait_streak",0))
    if active and ai_status in {"PENDING", "REJECTED", "DISCARDED", "WARMUP"}:
        block("AI_"+ai_status, {"PENDING":"Ожидается проверенный ответ Qwen",
              "REJECTED":"Последний ответ Qwen не принят", "DISCARDED":"Ответ Qwen отменён",
              "WARMUP":"Для Qwen ещё собираются наблюдения"}[ai_status])
    latest = next((dict(time=e["occurred_at"], action=e["payload"].get("action"),
                        reason=e["payload"].get("reason")) for e in reversed(events) if e["kind"] == "DECISION"), None)
    return dict(scope="ARCHIVE" if archived else "LIVE", recovery_required=recovery, as_of=stamp(reference),
                components=rows, blockers=blockers, latest_decision=latest)
