"""Три независимые виртуальные стратегии на общем потоке новых котировок."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

from .engine import Engine
from .models import D, encode, parse_time, stamp
from .qwen import prompt_version, QwenStrategy
from .strategy import Intent


class CashOnly:
    version = "usd-cash-v1"

    def decide(self, history, mid, has_position):
        return Intent("WAIT", "USD_BASELINE")


class Experiment:
    """Один сборщик снаружи, три Engine, ни одного брокерного исполнителя.

    Плечи сохраняются для анализа, но не восстанавливаются как обычные сессии:
    USD имеет отдельную стратегию, а исследовательский набор фиксируется до сбора.
    """

    @classmethod
    def start(cls, store, config, start, deadline, *, client=None):
        if config.source != "kraken-public":
            raise ValueError("Эксперимент поддерживает только kraken-public и локальные виртуальные средства")
        if not config.llm_digest:
            raise ValueError("Нужна заранее проверенная версия Qwen")
        if (deadline - start).total_seconds() <= config.flatten_buffer_seconds:
            raise ValueError("Эксперимент должен быть длиннее буфера завершения")
        result = cls()
        result.store, result.config = store, config
        result.started_at, result.deadline_at = stamp(start), stamp(deadline)
        result.engines, result.common_records, result.seen_steps = {}, [], set()
        result.closed, result.last_processed_at = False, None
        try:
            for role in ("usd", "demo_momentum", "qwen"):
                arm_config = replace(config, strategy_version="qwen-local-v1" if role == "qwen" else "demo-momentum-v1")
                engine = Engine.start(store, arm_config, start, deadline)
                result.engines[role] = engine
                store.save_analysis(engine.state.session_id, "experiment_role", dict(
                    schema_version="forward-experiment-role-v1", role=role, started_at=result.started_at,
                    deadline_at=result.deadline_at, dataset_role="forward_holdout", resume_allowed=False))
                if role == "usd":
                    engine.strategy = CashOnly()
                elif role == "qwen" and client is not None:
                    engine.strategy.close()
                    engine.strategy = QwenStrategy(client=client, model=config.llm_model)
        except Exception:
            result.close()
            raise
        return result

    def step(self, now, quote, *, error=None, stop=False, candle_context=None):
        if self.closed:
            raise ValueError("Эксперимент уже закрыт")
        processed = stamp(now)
        if now < parse_time(self.started_at) or (self.last_processed_at and now < parse_time(self.last_processed_at)):
            raise ValueError("Время эксперимента не может идти назад")
        raw = asdict(quote) if quote else None
        control = dict(processed_at=processed, quote=raw, error=error, stop=bool(stop))
        step_id = hashlib.sha256(encode(control).encode()).hexdigest()
        if step_id in self.seen_steps:
            return {}
        # Quote неизменяем; контекст копируется, чтобы сборщик не менял его при инференции.
        context = deepcopy(candle_context or {})
        generated = {}
        for role, engine in self.engines.items():
            engine.candle_provider = lambda when, frozen=context: deepcopy(frozen)
            generated[role] = engine.step(now, quote, error=error, stop=stop, step_id=step_id)
        self.seen_steps.add(step_id)
        self.last_processed_at = processed
        stable_quote = {k: str(v) for k, v in raw.items() if k != "quote_id"} if raw else None
        self.common_records.append(dict(processed_at=processed, quote=stable_quote, error=bool(error), stop=bool(stop),
            candle_context_sha256=hashlib.sha256(encode(context).encode()).hexdigest(),
            accepted_by={role: any(kind == "SNAPSHOT" for kind, _ in rows) for role, rows in generated.items()}))
        return generated

    def report(self):
        results, ids = {}, {}
        for role, engine in self.engines.items():
            report = engine.report()
            ids[role] = engine.state.session_id
            events = self.store.events(engine.state.session_id)
            fills = [e["payload"] for e in events if e["kind"] == "PAPER_FILL"]
            decisions = [e["payload"] for e in events if e["kind"] == "AI_DECISION"]
            requests = [e for e in events if e["kind"] == "AI_REQUEST"]
            accepted = sum(e.get("accepted") is True for e in decisions)
            latencies = [D(str(e["latency_ms"])) for e in decisions if e.get("latency_ms") is not None]
            results[role] = {key: report[key] for key in ("cash", "quantity", "status", "fees", "max_drawdown", "entries",
                "exits", "realized_pnl", "equity_at_last_quote", "valuation_observed_at", "exit_reason")}
            results[role].update(net_pnl=report["net_pnl_at_last_quote"], slippage_cost=report["modeled_slippage_cost"],
                turnover=str(sum((D(f["quantity"]) * D(f["price"]) for f in fills), D("0"))),
                simulated_order_count=len(fills),
                denied_risk_count=sum(e["kind"] == "RISK_VERDICT" and e["payload"].get("approved") is False for e in events),
                quote_count=sum(e["kind"] == "SNAPSHOT" for e in events),
                data_rejected_count=sum(e["kind"] == "DATA_REJECTED" for e in events),
                data_unavailable_count=sum(e["kind"] == "DATA_UNAVAILABLE" for e in events),
                ai_request_count=len(requests), ai_response_count=len(decisions), ai_accepted_count=accepted,
                ai_rejected_count=len(decisions) - accepted,
                ai_rejected_percent=str(D(len(decisions) - accepted) / len(decisions) * 100) if decisions else None,
                ai_response_coverage=str(D(len(decisions)) / len(requests)) if requests else None,
                mean_response_latency_ms=str(sum(latencies, D("0")) / len(latencies)) if latencies else None,
                max_response_latency_ms=str(max(latencies)) if latencies else None)
        quoted = [r for r in self.common_records if r["quote"]]
        all_closed = all(e.state.status == "CLOSED" for e in self.engines.values())
        return dict(schema_version="forward-experiment-v1", mode="LOCAL_FORWARD_SIMULATION",
            dataset_role="forward_holdout", started_at=self.started_at, planned_deadline_at=self.deadline_at,
            last_processed_at=self.last_processed_at, completed=all_closed, session_ids=ids,
            config=json.loads(encode(self.config.to_dict())), model=self.config.llm_model, model_digest=self.config.llm_digest,
            prompt_version=prompt_version(self.config.decision_policy,self.config.exit_policy,self.config.target_return),
            dataset_sha256=hashlib.sha256(encode(self.common_records).encode()).hexdigest(),
            common_quote_count=len(quoted), common_step_count=len(self.common_records),
            total_simulated_order_count=sum(row["simulated_order_count"] for row in results.values()),
            all_arms_accepted_quote_count=sum(all(r["accepted_by"].values()) for r in quoted),
            first_quote_processed_at=quoted[0]["processed_at"] if quoted else None,
            last_quote_processed_at=quoted[-1]["processed_at"] if quoted else None,
            results=results, evidence="UNASSESSED", resume_allowed=False,
            execution_model="immediate full virtual fills with configured costs",
            limitations=["Новые наблюдения разделены между тремя независимыми виртуальными портфелями.",
                "Название forward_holdout фиксируется до сбора; код не доказывает отсутствие прежнего знакомства с рынком.",
                "Короткая сессия и число ответов могут быть недостаточны для вывода о пользе Qwen.",
                "Комиссии и проскальзывание оценочные, реальные биржевые исполнения отсутствуют.",
                "При остатке оценка по последнему bid; результат не гарантирует прибыль или возможность закрытия."])

    def persist_report(self):
        report = self.report()
        for engine in self.engines.values():
            self.store.save_analysis(engine.state.session_id, "experiment", report)
        return report

    def close(self):
        if self.closed:
            return
        self.closed = True
        for engine in self.engines.values():
            if hasattr(engine.strategy, "close"):
                engine.strategy.close()
