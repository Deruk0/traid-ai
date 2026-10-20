"""Воспроизводимое сравнение сохранённой сессии без сети и инференции."""
from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

from .engine import Engine
from .models import Config, D, Quote, State, decimal, encode, parse_time, stamp
from .strategy import DemoMomentum, Intent
from .store import Store


def _hash(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def _stable(value):
    # Идентификаторы запусков не являются рыночными данными. Времена сохраняются.
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items()
                if k not in {"seq", "session_id", "quote_id", "decision_id", "snapshot_id"}}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


class _Cash:
    version = "usd-cash-v1"

    def decide(self, history, mid, has_position):
        return Intent("WAIT", "USD_BASELINE")


class RecordedStrategy:
    """Использует только ранее записанный ответ при неизменном контексте."""
    version = "recorded-qwen-v1"

    def __init__(self, config):
        self.original_config = config
        self.requests = {}
        self.quotes = {}
        self.decisions = []
        self.seen_answers = set()
        self.counts = Counter()

    def prepare(self, events, now):
        self.decisions = []
        for event in events:
            payload = event["payload"]
            if event["kind"] == "SNAPSHOT":
                self.quotes[payload.get("quote_id")] = dict(payload)
            elif event["kind"] == "AI_REQUEST":
                snapshot = payload.get("snapshot", {})
                snapshot_id = snapshot.get("snapshot_id")
                self.requests[snapshot_id] = (snapshot, parse_time(event["occurred_at"]),
                                              payload.get("model_digest"),
                                              self.quotes.get(snapshot.get("quote_id")),
                                              payload.get("model", "qwen3.5:4b"))
            elif event["kind"] == "AI_DECISION":
                self.counts["recorded_decisions"] += 1
                # Freeze the request visible BEFORE this answer, never a future request.
                self.decisions.append((payload, self.requests.get(payload.get("snapshot_id"))))

    def finish(self):
        self.counts["not_evaluated_by_strategy"] += len(self.decisions)
        self.decisions = []

    def decide_snapshot(self, config, state, quote, now):
        choices, self.decisions = self.decisions, []
        valid = []
        for payload, request in choices:
            try:
                action = self._validate(payload, request, state, quote, now)
            except (ValueError, TypeError, KeyError):
                self.counts["context_mismatches"] += 1
                continue
            self.counts["matched_decisions"] += 1
            valid.append(action)
        if len(valid) != 1:
            if len(valid) > 1:
                self.counts["ambiguous_decisions"] += len(valid)
            return Intent("WAIT", "NO_MATCHING_RECORDED_QWEN"), []
        self.counts["replayed_decisions"] += 1
        return Intent(valid[0], "RECORDED_QWEN_DECISION"), []

    def _validate(self, payload, request, state, quote, now):
        if (self.original_config.strategy_version != "qwen-local-v1"
                or payload.get("accepted") is not True or request is None):
            raise ValueError("Нет принятого ответа и предшествующего запроса")
        snapshot, requested_at, digest, recorded_quote, recorded_model = request
        snapshot_id = snapshot["snapshot_id"]
        if snapshot_id in self.seen_answers:
            raise ValueError("Повторный ответ")
        self.seen_answers.add(snapshot_id)
        body = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if _hash(body) != snapshot_id or payload["snapshot_id"] != snapshot_id:
            raise ValueError("Повреждённый снимок")
        if digest != self.original_config.llm_digest:
            raise ValueError("Другая версия модели")
        if (snapshot.get("model", "qwen3.5:4b") != self.original_config.llm_model
                or payload.get("model", "qwen3.5:4b") != self.original_config.llm_model
                or recorded_model != self.original_config.llm_model):
            raise ValueError("Другая модель сессии")
        from .qwen import inference_settings
        if snapshot.get("inference", inference_settings(False, self.original_config.llm_model)) != inference_settings(
                self.original_config.llm_thinking, self.original_config.llm_model):
            raise ValueError("Режим инференции не соответствует сессии")
        if parse_time(payload["processed_at"]) != now:
            raise ValueError("Другое время ответа")
        if not requested_at <= now or parse_time(snapshot["created_at"]) != requested_at:
            raise ValueError("Запрос из будущего")
        origin = Quote(snapshot.get("quote_id", ""), snapshot["observed_at"],
                       decimal(snapshot["market"]["bid"]), decimal(snapshot["market"]["ask"]),
                       snapshot["source"], snapshot["instrument"])
        origin.validate(now, self.original_config)
        if (not recorded_quote or recorded_quote.get("observed_at") != origin.observed_at
                or decimal(recorded_quote["bid"]) != origin.bid or decimal(recorded_quote["ask"]) != origin.ask
                or recorded_quote.get("source") != origin.source
                or recorded_quote.get("instrument", "BTC/USD") != origin.instrument):
            raise ValueError("Нет исходной сохранённой котировки")
        if quote.source != origin.source or quote.instrument != origin.instrument:
            raise ValueError("Другой рынок")
        portfolio = snapshot["portfolio"]
        for field in ("cash", "quantity", "entry_price", "stop_price"):
            if decimal(portfolio[field]) != decimal(getattr(state, field)):
                raise ValueError("Портфель изменился")
        if portfolio.get("entered_at") != state.entered_at:
            raise ValueError("Другая позиция")
        session = snapshot["session"]
        if session["status"] != state.status or parse_time(session["deadline_at"]) != parse_time(state.deadline_at):
            raise ValueError("Полномочия изменились")
        for field in ("capital", "risk_per_trade", "max_exposure", "loss_limit", "fee_rate", "slippage_bps"):
            if decimal(snapshot["policy"][field]) != getattr(self.original_config, field):
                raise ValueError("Лимиты изменились")
        answer = payload["answer"]
        evidence_contract = "decision_contract" in snapshot
        if (self.original_config.decision_policy in {"evidence-v1", "review-v1"}) != evidence_contract:
            raise ValueError("Версия контракта не соответствует политике сессии")
        if "decision_contract" in snapshot:
            versions = {"evidence-v1":{"market-summary-v4","market-summary-v5"},
                        "review-v1":{"market-summary-v6","market-summary-v7"}}
            if (snapshot.get("snapshot_version") not in versions.get(self.original_config.decision_policy,set())
                    or snapshot["policy"].get("decision_policy") != self.original_config.decision_policy):
                raise ValueError("Версия снимка не соответствует проверяемой политике")
            modern = self.original_config.exit_policy == "trailing-stop-v1" or self.original_config.target_return is not None
            if self.original_config.decision_policy == "review-v1" and modern != (snapshot.get("snapshot_version") == "market-summary-v7"):
                raise ValueError("Сопровождение/цель не соответствует версии снимка")
            if snapshot.get("snapshot_version") == "market-summary-v7":
                policy=snapshot["policy"]
                if "max_hold_seconds" in policy and policy["max_hold_seconds"] != self.original_config.max_hold_seconds:
                    raise ValueError("Максимум удержания изменился")
                if (policy.get("exit_policy") != self.original_config.exit_policy
                        or decimal(policy["stop_fraction"]) != self.original_config.stop_fraction
                        or (decimal(policy["target_return"]) if policy.get("target_return") is not None else None)
                            != self.original_config.target_return):
                    raise ValueError("Правило стопа или цель изменились")
            from .decision_context import validate_selection
            wire = {key:answer[key] for key in ("snapshot_id","action","reason_code","candidate_id","evidence_refs")}
            validate_selection(wire,snapshot)
        action = answer["action"]
        if (answer["snapshot_id"] != snapshot_id or payload.get("action") != action
                or action not in snapshot["allowed_actions"]
                or action not in {"WAIT", "REQUEST_ENTRY", "REQUEST_EXIT"}):
            raise ValueError("Недопустимое намерение")
        return action


class _ReplayEngine(Engine):
    retry_quote = None

    def _retry_strategy_quote(self, now):
        quote = self.state.quote()
        if not self.retry_quote or quote is None:
            return None
        if (self.retry_quote.get("quote_id") != quote.quote_id
                or self.retry_quote.get("observed_at") != quote.observed_at):
            return None
        try:
            quote.validate(now, self.config)
        except ValueError:
            return None
        return quote


def compare_session(export: dict) -> dict:
    """Сравнить полную сохранённую сессию на одинаковых виртуальных исполнениях.

    Экспорт соответствует /api/export: config, state, events. Исходные broker fills
    не воспроизводятся: все варианты исполняются локальным Engine, не BrokerEngine.
    """
    config = Config.from_dict(export["config"])
    initial = export["state"]
    started, deadline = parse_time(initial["started_at"]), parse_time(initial["deadline_at"])
    if deadline <= started:
        raise ValueError("Некорректный интервал сессии")
    events = export["events"]
    if not isinstance(events, list):
        raise ValueError("Нужен полный список событий")
    groups = []
    previous = started
    for event in events:
        now = parse_time(event["occurred_at"])
        if now < previous or not isinstance(event["payload"], dict):
            raise ValueError("События должны быть упорядочены по времени")
        previous = now
        if not groups or groups[-1][0] != now:
            groups.append((now, []))
        groups[-1][1].append(event)
    quotes = [e for e in events if e["kind"] == "SNAPSHOT"]
    if not quotes:
        raise ValueError("В экспорте отсутствуют котировки")
    # Один Engine.step может иметь только один принятый снимок.
    observed_before = None
    for now, rows in groups:
        if sum(e["kind"] == "SNAPSHOT" for e in rows) > 1:
            raise ValueError("Несколько котировок с одним временем обработки")
        for event in rows:
            if event["kind"] != "SNAPSHOT":
                continue
            raw = event["payload"]
            quote = Quote(raw["quote_id"], raw["observed_at"], decimal(raw["bid"]), decimal(raw["ask"]),
                          raw["source"], raw.get("instrument", "BTC/USD"))
            quote.validate(now, config)
            observed = parse_time(quote.observed_at)
            if observed_before is not None and observed <= observed_before:
                raise ValueError("Время наблюдений должно строго возрастать")
            observed_before = observed
    split = []
    boundaries = (0, len(quotes) * 60 // 100, len(quotes) * 80 // 100, len(quotes))
    for label, first, end in zip(("development", "validation", "holdout"), boundaries, boundaries[1:]):
        rows = quotes[first:end]
        split.append(dict(label=label, start_index=first, end_index_exclusive=end,
                          count=len(rows), first_processed_at=rows[0]["occurred_at"] if rows else None,
                          last_processed_at=rows[-1]["occurred_at"] if rows else None))
    replay = RecordedStrategy(config)
    results = {}
    offline_config = replace(config, strategy_version="demo-momentum-v1")
    with tempfile.TemporaryDirectory(prefix="traid-evaluation-") as directory:
        for name, strategy in (("usd", _Cash()), ("demo_momentum", DemoMomentum()), ("recorded_qwen", replay)):
            store = Store(Path(directory) / (name + ".sqlite3"))
            try:
                state = State(name, stamp(started), stamp(deadline), "RUNNING", str(config.capital), str(config.capital))
                store.create(offline_config, state)
                engine = _ReplayEngine(store, name)
                engine.strategy = strategy
                turnover, denied = D("0"), 0
                for index, (now, rows) in enumerate(groups):
                    engine.retry_quote = next((e["payload"] for e in rows if e["kind"] == "QUOTE_RECHECK"), None)
                    if strategy is replay:
                        replay.prepare(rows, now)
                    snapshot = next((e["payload"] for e in rows if e["kind"] == "SNAPSHOT"), None)
                    quote = None
                    if snapshot:
                        quote = Quote(snapshot["quote_id"], snapshot["observed_at"], decimal(snapshot["bid"]),
                                      decimal(snapshot["ask"]), snapshot["source"], snapshot.get("instrument", "BTC/USD"))
                        quote.validate(now, config)
                    pause, stop, stop_reason = None, False, None
                    for event in rows:
                        if event["kind"] == "WATCHDOG_TRIPPED":
                            stop, stop_reason = True, event["payload"].get("reason", "WORKER_STALLED")
                        if event["kind"] == "SESSION_STATUS":
                            status = event["payload"].get("to")
                            if status in {"PAUSED_ENTRIES", "RUNNING"}:
                                pause = status == "PAUSED_ENTRIES"
                            reason = event["payload"].get("reason")
                            if reason in {"USER_STOP", "WORKER_STALLED", "SYSTEM_RESUME_GAP", "WORKER_ERROR"}:
                                stop, stop_reason = True, reason
                    if stop_reason and not engine.state.exit_reason:
                        engine.state.exit_reason = stop_reason
                        store.commit_step(engine.state, "control-" + str(index), stamp(now), [])
                    generated = engine.step(now, quote, step_id=str(index), pause_entries=pause, stop=stop)
                    for kind, payload in generated:
                        if kind == "PAPER_FILL":
                            turnover += decimal(payload["quantity"]) * decimal(payload["price"])
                        if kind == "RISK_VERDICT" and payload.get("approved") is False:
                            denied += 1
                    if strategy is replay:
                        replay.finish()
                report = engine.report()
                results[name] = {k: report[k] for k in ("cash", "quantity", "status", "fees", "entries", "exits",
                    "max_drawdown", "realized_pnl", "equity_at_last_quote", "valuation_observed_at", "exit_reason")}
                results[name].update(net_pnl=report["net_pnl_at_last_quote"], turnover=str(turnover),
                                     denied_risk_count=denied, slippage_cost=report["modeled_slippage_cost"])
            finally:
                store.close()
    counts = {key: replay.counts[key] for key in ("recorded_decisions", "matched_decisions", "replayed_decisions",
              "context_mismatches", "not_evaluated_by_strategy", "ambiguous_decisions")}
    counts["coverage"] = str(D(counts["replayed_decisions"]) / counts["recorded_decisions"]) if counts["recorded_decisions"] else "0"
    return dict(schema_version="session-comparison-v1", mode="OFFLINE_SIMULATION", source=config.source,
        dataset_sha256=_hash(_stable(dict(config=export["config"], events=events,
                                         started_at=stamp(started), deadline_at=stamp(deadline)))),
        observation_count=len(quotes), results=results, qwen_replay=counts, temporal_split=split,
        evidence="NO_VERIFIED_ADVANTAGE", execution_model="immediate full virtual fills with configured costs",
        limitations=["Расчёт полной сессии; временные части — метки данных, отдельной проверки holdout нет.",
                     "Qwen: только записанные ответы с совпадающим портфелем; это не новая независимая инференция.",
                     "Реальные исполнения и поздние комиссии Alpaca здесь не моделируются.",
                     "Подбор на известной истории создаёт риск знания будущего; нужны новые независимые forward-сессии.",
                     "Остаток оценивается по последнему bid; это не обещание закрытия или прибыли."])
