"""Research candidates and code-verified explanations; no risk permissions."""
from datetime import timedelta
from .models import D, decimal, parse_time, stamp

ENTRY_ID = "entry-candle-momentum-v1"
REVIEW_ENTRY_ID = "entry-momentum-review-v1"
EXIT_ID = "exit-position-review-v1"
HOLDING_SECONDS = 300
WIRE_KEYS = {"snapshot_id", "action", "reason_code", "candidate_id", "evidence_refs"}


def _numeric(value):
    return str(decimal(value)) if value is not None else None


def _display(value):
    if value is None:
        return "нет данных"
    if isinstance(value, bool):
        return "да" if value else "нет"
    try:
        number = decimal(value)
        if number == 0:
            return "0"
        if abs(number) >= D("1e12") or (number and abs(number) < D("1e-8")):
            return str(number)
        return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")
    except ValueError:
        return str(value)


def build_decision_context(config, state, quote, now, analytics, candles):
    """The explicit entry rule is a research filter, not expected trading profit."""
    if candles.get("source", quote.source) != quote.source or candles.get("instrument", quote.instrument) != quote.instrument:
        raise ValueError("Источники контекста не совпадают")
    windows = {w["horizon_minutes"]: w for w in candles.get("windows", [])}
    c5, c15 = windows.get(5, {}), windows.get(15, {})
    candle_status = candles.get("status", "UNAVAILABLE")
    count = int(candles.get("count", 0))
    ready5, ready15 = c5.get("status") == "READY", c15.get("status") == "READY"
    volume = decimal(c5["volume"]) if ready5 and c5.get("volume") is not None else None
    if volume is not None and volume < 0:
        raise ValueError("Объём не может быть отрицательным")
    volume_state = ("STALE" if candle_status == "STALE" else "UNAVAILABLE" if not count else
                    "INCOMPLETE" if not ready5 else "UNAVAILABLE" if volume is None else
                    "ZERO" if volume == 0 else "AVAILABLE")
    quality = dict(candle_status=candle_status, candle_count=count,
                   usable_horizons_minutes=sorted(m for m, w in windows.items() if w.get("status") == "READY"),
                   candles_available=count > 0,
                   volume_available=any(w.get("status") == "READY" and w.get("volume") is not None for w in windows.values()),
                   volume_state=volume_state, volume_window_minutes=5, quote_data_kind="sampled_bid_ask",
                   source=quote.source, instrument=quote.instrument)
    cost = decimal(analytics["costs"]["estimated_round_trip_bps"])/100
    if cost < 0:
        raise ValueError("Расходы не могут быть отрицательными")
    seconds = max(D(0), decimal((parse_time(state.deadline_at)-now).total_seconds())-config.flatten_buffer_seconds)
    qty = decimal(state.quantity)
    r5 = decimal(c5["return_pct"]) if ready5 else None
    r15 = decimal(c15["return_pct"]) if ready15 else None
    range15 = decimal(c15["range_pct"]) if ready15 else None
    evidence = dict(candle_status=candle_status, c5_status=c5.get("status", "UNAVAILABLE"),
                    c15_status=c15.get("status", "UNAVAILABLE"), c5_return_pct=_numeric(r5),
                    c15_return_pct=_numeric(r15), c5_volume_btc=_numeric(volume), c15_range_pct=_numeric(range15),
                    round_trip_cost_pct=str(cost), seconds_to_flatten=str(seconds), position_qty=str(qty),
                    candle_count=str(count), session_status=state.status,
                    pending_order=bool((state.broker or {}).get("pending")))
    def condition(passed, *refs):
        return dict(passed=bool(passed), evidence_refs=list(refs))
    conditions = dict(
        candles_ready=condition(ready5 and ready15, "c5_status", "c15_status"),
        positive_trend=condition(ready5 and ready15 and r5 > 0 and r15 > 0, "c5_return_pct", "c15_return_pct"),
        positive_volume=condition(volume is not None and volume > 0, "c5_status", "c5_volume_btc"),
        cost_scale=condition(ready15 and range15 > cost, "c15_range_pct", "round_trip_cost_pct"),
        entry_window=condition(seconds >= HOLDING_SECONDS, "seconds_to_flatten"),
        entries_enabled=condition(qty == 0 and state.status == "RUNNING" and not evidence["pending_order"],
                                  "position_qty", "session_status", "pending_order"))
    eligible = all(c["passed"] for c in conditions.values())
    review = config.decision_policy == "review-v1"
    entry_id = REVIEW_ENTRY_ID if review else ENTRY_ID
    if review:
        for name, item in conditions.items():
            item["required"] = name != "cost_scale"
        eligible = all(c["passed"] for c in conditions.values() if c["required"])
    candidates = [dict(candidate_id=entry_id, action="REQUEST_ENTRY", research_only=True,
                       status="NOT_APPLICABLE" if qty > 0 else "ELIGIBLE" if eligible else "BLOCKED",
                       expected_return=None, holding_horizon_seconds=HOLDING_SECONDS,
                       conditions=conditions, invalidation=[name for name,c in conditions.items()
                                                          if c.get("required", True)],
                       cost_scale_basis="historical_range_not_expected_return_or_proven_edge")]
    contract = []
    def option(action, code, *refs, candidate_id=entry_id):
        contract.append(dict(action=action, reason_code=code, candidate_id=candidate_id,
                             evidence_refs=list(refs)))
    if qty > 0:
        candidates.append(dict(candidate_id=EXIT_ID, action="REQUEST_EXIT", research_only=True,
                               status="ELIGIBLE", expected_return=None, holding_horizon_seconds=None,
                               conditions=dict(position_exists=condition(True, "position_qty")),
                               invalidation=["position_exists"]))
        option("WAIT", "POSITION_MONITORING", "position_qty", candidate_id=EXIT_ID)
        if config.exit_policy != "trailing-stop-v1":
            option("REQUEST_EXIT", "POSITION_REVIEW", "position_qty", candidate_id=EXIT_ID)
        else:
            candidates[-1]["status"]="INDEPENDENT_TRAILING_CONTROL"
    else:
        if not count:
            option("WAIT", "CANDLES_UNAVAILABLE", "candle_count", "candle_status")
        elif candle_status == "STALE":
            option("WAIT", "CANDLES_STALE", "candle_status")
        elif not (ready5 and ready15):
            option("WAIT", "CANDLES_INCOMPLETE", "c5_status", "c15_status")
        if ready5 and volume is None:
            option("WAIT", "VOLUME_UNAVAILABLE", "c5_status", "c5_volume_btc")
        if ready5 and volume == 0:
            option("WAIT", "ZERO_VOLUME", "c5_status", "c5_volume_btc")
        if ready5 and ready15 and (r5 <= 0 or r15 <= 0):
            option("WAIT", "TREND_NOT_CONFIRMED", "c5_return_pct", "c15_return_pct")
        if ready15 and range15 <= cost:
            option("WAIT", "COST_SCALE_TOO_SMALL", "c15_range_pct", "round_trip_cost_pct")
        if seconds < HOLDING_SECONDS:
            option("WAIT", "ENTRY_WINDOW_TOO_SHORT", "seconds_to_flatten")
        if state.status != "RUNNING" or evidence["pending_order"]:
            option("WAIT", "ENTRIES_DISABLED", "session_status", "pending_order")
        if eligible:
            positive_refs = list(dict.fromkeys(ref for c in conditions.values() for ref in c["evidence_refs"]))
            option("REQUEST_ENTRY", "ENTRY_REVIEW_ACCEPTED" if review else "ENTRY_CRITERIA_MET", *positive_refs)
            option("WAIT", "NEXT_CANDLE_CONFIRMATION" if review else "CANDIDATE_DECLINED", *positive_refs)
    if not contract:
        raise ValueError("Контекст не позволяет построить доказуемую причину решения")
    result = dict(data_quality=quality, evidence=evidence, candidates=candidates, decision_contract=contract)
    if review:
        # The clock is owned by code. Reassessment never becomes an order by itself.
        review_at = min(now+timedelta(seconds=60),
                        parse_time(state.deadline_at)-timedelta(seconds=config.flatten_buffer_seconds))
        result["review_policy"] = dict(version="wait-review-v1", review_at=stamp(max(now,review_at)),
            trigger="NEXT_CLOSED_CANDLE_OR_TIMER", candle_end_at=candles.get("latest_end_at"),
            automatic_entry=False)
    return result


def render_reason(reason_code, evidence):
    """Human text is always rendered by code from the selected factual option."""
    e = {key: _display(value) for key, value in evidence.items()}
    templates = {
        "CANDLES_UNAVAILABLE": "Свечной истории нет: доступно {candle_count} свечей.",
        "CANDLES_STALE": "Свечная история устарела: статус {candle_status}.",
        "CANDLES_INCOMPLETE": "Недостаточно полных свечных окон: 5 мин — {c5_status}, 15 мин — {c15_status}.",
        "VOLUME_UNAVAILABLE": "Окно 5 мин готово, но измеренный объём недоступен.",
        "ZERO_VOLUME": "Объём за готовое окно 5 мин равен {c5_volume_btc} BTC; условие входа не выполнено.",
        "TREND_NOT_CONFIRMED": "Положительное движение на обоих окнах не подтверждено: 5 мин {c5_return_pct} %, 15 мин {c15_return_pct} %.",
        "COST_SCALE_TOO_SMALL": "Диапазон за 15 мин {c15_range_pct} % не превышает оценку расходов {round_trip_cost_pct} %. Это исторический фильтр.",
        "ENTRY_WINDOW_TOO_SHORT": "До начала закрытия осталось {seconds_to_flatten} с; исследовательскому кандидату требуется не менее 300 с.",
        "ENTRIES_DISABLED": "Новые входы недоступны: состояние {session_status}, ожидающая заявка — {pending_order}.",
        "ENTRY_CRITERIA_MET": "Условия исследовательского кандидата выполнены: движение 5/15 мин {c5_return_pct}/{c15_return_pct} %, объём {c5_volume_btc} BTC. Требуется проверка риска.",
        "CANDIDATE_DECLINED": "Условия исследовательского кандидата выполнены, но модель выбрала ожидание. Движение 5/15 мин {c5_return_pct}/{c15_return_pct} %, объём {c5_volume_btc} BTC.",
        "ENTRY_REVIEW_ACCEPTED": "Модель предложила вход: движение 5/15 мин {c5_return_pct}/{c15_return_pct} %, объём {c5_volume_btc} BTC; расходы {round_trip_cost_pct} %. Требуется разрешение риска.",
        "NEXT_CANDLE_CONFIRMATION": "Кандидат доступен: движение 5/15 мин {c5_return_pct}/{c15_return_pct} %, расходы {round_trip_cost_pct} %. Модель запросила подтверждение следующей закрытой свечой.",
        "POSITION_MONITORING": "Модель выбрала наблюдение за открытой позицией {position_qty} BTC; независимые стопы и дедлайн продолжают действовать.",
        "POSITION_REVIEW": "Модель предложила выход из позиции {position_qty} BTC; исполнение проверяет независимый код.",
    }
    if reason_code not in templates:
        raise ValueError("Неизвестная проверяемая причина")
    try:
        return templates[reason_code].format(**e)[:240]
    except KeyError as exc:
        raise ValueError("Нет фактов для объяснения") from exc


def validate_selection(answer, snapshot):
    if snapshot.get("snapshot_version") in {"market-summary-v6","market-summary-v7"} and not isinstance(snapshot.get("review_policy"),dict):
        raise ValueError("В новом контракте отсутствует политика пересмотра")
    if (not isinstance(answer, dict) or set(answer) != WIRE_KEYS
            or not isinstance(answer["snapshot_id"], str) or not answer["snapshot_id"]
            or answer["snapshot_id"] != snapshot.get("snapshot_id")
            or not isinstance(answer["action"], str) or answer["action"] not in snapshot.get("allowed_actions", [])
            or not isinstance(answer["reason_code"], str) or not isinstance(answer["candidate_id"], str)
            or not isinstance(answer["evidence_refs"], list)
            or any(not isinstance(ref, str) for ref in answer["evidence_refs"])):
        raise ValueError("Ответ не соответствует контракту решения")
    refs = answer["evidence_refs"]
    evidence = snapshot.get("evidence", {})
    if len(set(refs)) != len(refs) or any(ref not in evidence for ref in refs):
        raise ValueError("Некорректные ссылки на факты")
    for option in snapshot.get("decision_contract", []):
        if (all(answer[key] == option.get(key) for key in ("action", "reason_code", "candidate_id"))
                and set(refs) == set(option.get("evidence_refs", []))):
            result = dict(answer, evidence_refs=list(option["evidence_refs"]),
                          reason=render_reason(answer["reason_code"], evidence), explanation_status="VERIFIED")
            if snapshot.get("review_policy") and answer["action"] == "WAIT":
                policy = snapshot["review_policy"]
                at, created = parse_time(policy["review_at"]), parse_time(snapshot["created_at"])
                if (policy.get("version") != "wait-review-v1" or policy.get("automatic_entry") is not False
                        or policy.get("trigger") != "NEXT_CLOSED_CANDLE_OR_TIMER"
                        or not 0 <= (at-created).total_seconds() <= 60):
                    raise ValueError("Некорректный срок пересмотра ожидания")
                result.update(review_at=policy["review_at"],review_trigger=policy["trigger"],
                              review_candle_end=policy.get("candle_end_at"))
                if answer["reason_code"] == "COST_SCALE_TOO_SMALL":
                    result["reason"] = result["reason"].replace("Это исторический фильтр.",
                        "Модель выбрала пересмотр с учётом расходов.")[:240]
            return result
    raise ValueError("Выбранная причина не подтверждена фактами этого снимка")
