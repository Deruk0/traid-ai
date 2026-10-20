"""Summaries of recorded decisions; never recompute historical trading permissions."""
from collections import Counter
from .models import decimal


IMPORTANT_EVENTS = frozenset({
    "PAPER_FILL", "BROKER_FILL", "SESSION_STARTED", "SESSION_STATUS",
    "DATA_UNAVAILABLE", "DATA_REJECTED", "DECISION_UNAVAILABLE", "WORKER_ERROR",
    "WATCHDOG_TRIPPED", "ORDER_PREPARED", "ORDER_UPDATE", "ORDER_UNKNOWN",
    "ORDER_REJECTED", "BROKER_UNRESOLVED", "ORDER_CANCEL_REQUESTED",
    "ORDER_ABORTED", "ORDER_ABSENT_CONFIRMED",
    "BROKER_RECONCILIATION_MISMATCH", "LATE_FEES_RECONCILED", "ACCOUNTING_UNRESOLVED",
})

CONDITIONS = {
    "candles_ready": "Свечные окна ещё не готовы",
    "positive_trend": "Положительное изменение за 5 и 15 минут не подтверждено",
    "positive_volume": "Нет положительного измеренного объёма за 5 минут",
    "cost_scale": "Диапазон за 15 минут не превышает оценку расходов",
    "entry_window": "До закрытия осталось меньше времени, чем требует кандидат",
    "entries_enabled": "Вход запрещён состоянием сессии, позицией или ожидающей заявкой",
}

WAIT_REASONS = {
    "WARMUP": ("Стратегия", "Недостаточно наблюдений для решения"),
    "NO_SIGNAL": ("Стратегия", "Условия сигнала не выполнены"),
    "AI_PENDING": ("Qwen", "Ожидается ответ модели"),
    "AI_COOLDOWN": ("Планировщик Qwen", "Пауза между запросами модели"),
    "AI_REVIEW_SCHEDULED": ("Планировщик Qwen", "Ожидается срок пересмотра или новая закрытая свеча"),
    "BROKER_RECONCILING": ("Брокер / сверка", "Новая заявка ждёт сверки брокера"),
    "ENTRIES_PAUSED": ("Управление сессией", "Пользователь приостановил входы"),
    "TRAILING_POSITION_HOLD": ("Скользящий стоп", "Штатное удержание: защитный выход ещё не сработал"),
    "DECISION_UNAVAILABLE": ("Стратегия", "Не удалось получить решение"),
}


def blocking_log(events, limit=200):
    """Project recorded blockers and waits, without inventing permission checks.

    Infer direction only from a linked intention or recorded position. Unknown
    direction stays explicit. Repeated adjacent reasons share one display row;
    original events remain intact in the journal and export.
    """
    decisions = {e["payload"].get("decision_id"): e["payload"].get("action")
                 for e in events if e["kind"] == "DECISION" and e["payload"].get("decision_id")}
    requests, rows = {}, []
    position = None

    def add(event, scope, component, code, detail, *, status="Блокировка", facts=None):
        payload = dict(scope=scope, component=component, code=code, detail=detail,
                       status=status, source_kind=event["kind"], source_seq=event.get("seq"),
                       decision_id=event["payload"].get("decision_id"), facts=facts or {}, count=1,
                       first_at=event["occurred_at"])
        key = (scope, component, code, detail, status)
        if rows and rows[-1]["key"] == key:
            payload["count"] = rows[-1]["payload"]["count"] + 1
            payload["first_at"] = rows[-1]["payload"]["first_at"]
            rows.pop()
        rows.append(dict(seq=f"{event.get('seq', len(rows))}:{code}",
                         kind="TRADE_BLOCKER", occurred_at=event["occurred_at"], payload=payload, key=key))

    for event in events:
        kind, p = event["kind"], event["payload"]
        action = p.get("action") or decisions.get(p.get("decision_id"))
        scope = {"REQUEST_ENTRY": "Вход", "REQUEST_EXIT": "Выход"}.get(action, "Вход / выход")
        if kind == "SESSION_STARTED":
            position = False
        elif kind in {"BROKER_FILL", "PAPER_FILL"}:
            # A SELL can be partial. Only a ledger or a known zero proves flatness.
            position = True if p.get("side", "").upper() == "BUY" else False if kind == "PAPER_FILL" else None
            if rows:
                rows[-1]["key"] = None
        elif kind == "BROKER_LEDGER":
            try:
                position = decimal(p["quantity"]) > 0
            except (KeyError, ValueError, TypeError):
                position = None
        elif kind == "AI_REQUEST":
            snapshot = p.get("snapshot", {})
            requests[snapshot.get("snapshot_id")] = snapshot
            for candidate in snapshot.get("candidates", []):
                if candidate.get("action") != "REQUEST_ENTRY":
                    continue
                for name, condition in candidate.get("conditions", {}).items():
                    if condition.get("passed") is False and condition.get("required", True):
                        facts = {ref: snapshot.get("evidence", {}).get(ref)
                                 for ref in condition.get("evidence_refs", [])}
                        add(event, "Вход", "Фильтры кандидата", name,
                            CONDITIONS.get(name, name), facts=facts)
        elif kind == "AI_DECISION":
            snapshot = requests.get(p.get("snapshot_id"), {})
            if p.get("accepted") is False:
                add(event, "Решение ИИ", "Проверка ответа Qwen", p.get("error_code") or "AI_REJECTED",
                    p.get("reason") or "Ответ модели не принят")
            elif p.get("action") == "WAIT":
                contract = snapshot.get("decision_contract")
                allowed = ([o.get("action") for o in contract] if contract is not None
                           else snapshot.get("allowed_actions", []))
                if "REQUEST_ENTRY" in allowed or "REQUEST_EXIT" in allowed:
                    add(event, "Вход" if "REQUEST_ENTRY" in allowed else "Выход", "Qwen",
                        p.get("reason_code") or "MODEL_WAIT", p.get("reason") or "Модель выбрала ожидание",
                        status="Ожидание модели")
        elif kind == "DECISION" and p.get("action") == "WAIT" and p.get("reason") in WAIT_REASONS:
            code = p["reason"]
            component, detail = WAIT_REASONS[code]
            wait_scope = ("Выход" if code == "TRAILING_POSITION_HOLD" else "Вход" if code == "ENTRIES_PAUSED"
                          or position is False else "Выход" if position is True else "Вход / выход")
            add(event, wait_scope, component, code, detail, status="Штатное удержание" if
                code == "TRAILING_POSITION_HOLD" else "Ожидание")
        elif kind == "RISK_VERDICT" and p.get("approved") is False:
            add(event, scope, "Брокер / сверка" if p.get("reason") == "BROKER_NOT_READY" else "Risk Engine",
                p.get("reason") or "RISK_DENIED", p.get("reason") or "Заявка не разрешена")
        elif kind in {"BROKER_UNRESOLVED", "BROKER_RECONCILIATION_MISMATCH", "ORDER_UNKNOWN",
                      "ORDER_REJECTED", "ORDER_ABORTED", "ACCOUNTING_UNRESOLVED"}:
            order_scope = ("Вход" if kind == "ORDER_ABORTED" else
                           {"buy": "Вход", "sell": "Выход"}.get(str(p.get("side", "")).lower(), scope))
            add(event, order_scope, "Исполнитель / брокер", p.get("reason") or kind,
                p.get("reason") or ("Начисления не сверены" if kind == "ACCOUNTING_UNRESOLVED"
                                    else "Исполнение или баланс требуют подтверждения"),
                facts={k: p[k] for k in ("actual_cash", "actual_quantity", "expected_cash_low",
                    "expected_cash_high", "reason_codes") if k in p})
        elif kind in {"DATA_UNAVAILABLE", "DATA_REJECTED", "WATCHDOG_TRIPPED", "WORKER_ERROR"}:
            add(event, "Контекст решения" if kind.startswith("DATA_") else "Вход / выход",
                "Данные" if kind.startswith("DATA_") else "Контроль процесса",
                kind, p.get("reason") or "Подробности в исходном событии",
                status="Проблема данных" if kind.startswith("DATA_") else "Блокировка")
        elif kind == "PROTECTIVE_EXIT_REQUIRED":
            add(event, "Выход", "Данные / исполнение", "EXIT_QUOTE_REQUIRED",
                "Защитный выход отмечен без принятой котировки; результат исполнения записывается отдельно",
                facts={"exit_reason": p.get("reason")}, status="Требуется выход")
    selected = rows if limit is None else rows[-limit:]
    return [{k: v for k, v in row.items() if k != "key"} for row in selected]


def decision_summary(events):
    counts, reasons, blocked, warnings, latest_facts = Counter(), Counter(), Counter(), Counter(), {}
    risk_reasons, rejected_reasons = Counter(), Counter()
    requests, answers = set(), set()
    for event in events:
        p = event["payload"]
        if event["kind"] == "AI_REQUEST":
            snapshot = p.get("snapshot", {})
            requests.add(snapshot.get("snapshot_id"))
            counts["requests"] += 1
            choices = snapshot.get("decision_contract")
            if (any(o.get("action") == "REQUEST_ENTRY" for o in choices) if choices is not None
                    else "REQUEST_ENTRY" in snapshot.get("allowed_actions", [])):
                counts["requests_with_entry_option"] += 1
            for candidate in snapshot.get("candidates", []):
                if candidate.get("action") != "REQUEST_ENTRY":
                    continue
                status = candidate.get("status")
                counts[{"ELIGIBLE":"eligible", "BLOCKED":"blocked", "NOT_APPLICABLE":"not_applicable"}.get(status,"unknown")] += 1
                if status in {"BLOCKED","ELIGIBLE"}:
                    for name, condition in candidate.get("conditions", {}).items():
                        if condition.get("passed") is False:
                            (blocked if condition.get("required",True) else warnings)[name] += 1
                            latest_facts[name] = {ref:snapshot.get("evidence", {}).get(ref)
                                                  for ref in condition.get("evidence_refs", [])}
        elif event["kind"] == "AI_DECISION":
            answers.add(p.get("snapshot_id"))
            counts["answers"] += 1
            if p.get("accepted"):
                counts["accepted"] += 1
                counts["accepted_waits" if p.get("action") == "WAIT" else "accepted_entries" if
                       p.get("action") == "REQUEST_ENTRY" else "accepted_exits"] += 1
                if p.get("reason_code"):
                    reasons[p["reason_code"]] += 1
            else:
                counts["rejected"] += 1
                rejected_reasons[p.get("reason") or "UNKNOWN"] += 1
        elif event["kind"] == "RISK_VERDICT" and p.get("approved") is False:
            counts["risk_denials"] += 1
            risk_reasons[p.get("reason") or "UNKNOWN"] += 1
        elif event["kind"] in {"PAPER_FILL", "BROKER_FILL"}:
            counts["fills"] += 1
    return dict(**{key:counts[key] for key in ("requests", "answers", "accepted", "rejected",
        "accepted_waits", "accepted_entries", "accepted_exits", "requests_with_entry_option",
        "eligible", "blocked", "not_applicable", "risk_denials", "fills")},
        unanswered=len((requests-answers)-{None}), reason_counts=dict(reasons),
        risk_reason_counts=dict(risk_reasons), rejected_reason_counts=dict(rejected_reasons),
        blocking_conditions=[dict(code=name, label=CONDITIONS.get(name, name), count=count,
                                  latest_facts=latest_facts[name]) for name,count in blocked.most_common()],
        warnings=[dict(code=name,label=CONDITIONS.get(name,name),count=count,
                       latest_facts=latest_facts[name]) for name,count in warnings.most_common()])
