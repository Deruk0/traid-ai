import argparse
import math
from datetime import timedelta
from pathlib import Path
import sqlite3
import sys
import time
import json
import threading

from .engine import Engine
from .feeds import public_quote, synthetic_quote
from .models import Config, D, decimal, encode, parse_time, stamp, utc_now
from .store import Store, writer_lock


def settings(args, source: str) -> Config:
    hold_seconds = decimal(args.max_hold_minutes)*60 if args.max_hold_minutes is not None else None
    if hold_seconds is not None and (hold_seconds <= 0 or hold_seconds != hold_seconds.to_integral_value()):
        raise ValueError("Максимум удержания должен задавать положительное целое число секунд")
    return Config(source=source, capital=decimal(args.capital),
                  risk_per_trade=decimal(args.risk_percent) / 100,
                  max_exposure=decimal(args.exposure_percent) / 100,
                  loss_limit=decimal(args.loss_limit_percent) / 100,
                  fee_rate=decimal(args.fee_percent) / 100,
                  slippage_bps=decimal(args.slippage_bps),
                  stop_fraction=decimal(args.stop_percent)/100,
                  target_return=decimal(args.target_percent)/100 if args.target_percent is not None else None,
                  max_hold_seconds=int(hold_seconds) if hold_seconds is not None else None)


def show_events(events):
    for kind, payload in events:
        if kind in {"PAPER_FILL", "SESSION_STATUS", "DATA_UNAVAILABLE", "DATA_REJECTED"}:
            print(f"{kind}: {encode(payload)}", flush=True)


def run_demo(engine: Engine):
    start = parse_time(engine.state.started_at)
    deadline = parse_time(engine.state.deadline_at)
    duration = (deadline - start).total_seconds()
    last = parse_time(engine.state.last_tick_at) if engine.state.last_tick_at else None
    samples = max(60, math.ceil(duration / 10))
    for index in range(samples + 1):
        now = start + timedelta(seconds=duration * index / samples)
        if last and now <= last:
            continue
        show_events(engine.step(now, synthetic_quote(index, now)))
        if engine.state.status == "CLOSED":
            break


def run_public(engine: Engine, interval: int):
    print("PAPER: публичные BTC/USD, только виртуальные сделки. Ctrl+C — завершение.", flush=True)
    from .runtime import QuoteFeed
    from .watchdog import Watchdog
    from .history import CandleFeed, load_kraken_candles
    wake, trip = threading.Event(), []
    if hasattr(engine.strategy, "wake"):
        engine.strategy.wake = wake
    def on_trip(reason):
        trip.append(reason)
        wake.set()
    watchdog = Watchdog(on_trip)
    quotes = QuoteFeed(public_quote, interval_seconds=interval)
    candles = CandleFeed(load_kraken_candles, "kraken-public")
    if engine.config.strategy_version == "qwen-local-v1":
        engine.candle_provider = candles.view
        candles.refresh(utc_now())
    watchdog.start()
    quotes.refresh()
    try:
        while engine.state.status != "CLOSED":
            wake.clear()
            quote, error = quotes.take()
            quotes.refresh()
            now = utc_now()
            if engine.config.strategy_version == "qwen-local-v1":
                candles.refresh(now)
            if trip:
                engine.config, engine.state = engine.store.load(engine.state.session_id)
                engine.state.exit_reason = trip[0]
                engine.store.commit_step(engine.state,"watchdog:"+stamp(now),stamp(now),[("WATCHDOG_TRIPPED",dict(reason=trip[0]))])
            show_events(engine.step(now, quote, error=error, stop=bool(trip)))
            watchdog.progress()
            if engine.state.status in {"CLOSED", "UNRESOLVED"}:
                break
            # Wake for deadline handling even when the configured poll is longer.
            deadline = parse_time(engine.state.deadline_at)
            flatten_at = deadline - timedelta(seconds=engine.config.flatten_buffer_seconds)
            wake_at = flatten_at if now < flatten_at else deadline
            remaining = (wake_at - now).total_seconds()
            wake.wait(max(0, min(10, remaining)))
    except KeyboardInterrupt:
        # Never pretend that a stale mark is an executable exit quote.
        show_events(engine.step(utc_now(), None, stop=True))
        if engine.state.status != "CLOSED":
            print("Выход не подтверждён: выполните resume при доступных котировках.", flush=True)
    finally:
        quotes.close()
        candles.close()
        watchdog.stop()
        if hasattr(engine.strategy,"close"):
            engine.strategy.close()


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Traid AI: симулятор, реальные сделки отсутствуют")
    commands = cli.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser("dashboard", help="Открыть локальную панель управления")
    dashboard.add_argument("--db", type=Path, default=Path("data/paper.sqlite3"))
    dashboard.add_argument("--port", type=int, default=8787)
    dashboard.add_argument("--open", action="store_true", help="Открыть панель в браузере")
    compare = commands.add_parser("compare", help="Сравнить закрытую сессию: USD, правило, записанная Qwen")
    compare.add_argument("session_id")
    compare.add_argument("--db", type=Path, default=Path("data/paper.sqlite3"))
    compare.add_argument("--output", type=Path)
    experiment = commands.add_parser("experiment", help="Независимый forward: USD, правило, настоящая Qwen, локальная симуляция")
    experiment.add_argument("--db", type=Path, default=Path("data/experiments.sqlite3"))
    experiment.add_argument("--duration-minutes", default="10")
    experiment.add_argument("--capital", default="1000")
    experiment.add_argument("--output", type=Path, default=Path("data/forward-comparison.json"))
    for name in ("demo", "paper"):
        command = commands.add_parser(name, help="Синтетический прогон" if name == "demo" else "Публичные котировки")
        command.add_argument("--db", type=Path, default=Path("data/paper.sqlite3"))
        command.add_argument("--capital", default="1000", help="Виртуальный бюджет USD")
        command.add_argument("--risk-percent", default="0.5", help="Расчётный риск одной сделки, %%")
        command.add_argument("--exposure-percent", default="25", help="Максимальная экспозиция, %%")
        command.add_argument("--loss-limit-percent", default="3", help="Лимит потери сессии, %%")
        command.add_argument("--fee-percent", default="0.1", help="Моделируемая комиссия каждой стороны, %%")
        command.add_argument("--slippage-bps", default="5", help="Моделируемое проскальзывание, bps")
        command.add_argument("--stop-percent",default="2",help="Расстояние скользящего стопа, %%")
        command.add_argument("--target-percent",help="Желаемый результат к завершению, %%; мягкая цель")
        command.add_argument("--max-hold-minutes",help="Необязательный максимум удержания позиции; по умолчанию отключён")
        command.add_argument("--duration-minutes", default="60" if name == "demo" else "240")
        if name == "paper":
            command.add_argument("--interval-seconds", type=int, default=10)
    for name in ("report", "resume"):
        command = commands.add_parser(name, help="Отчёт" if name == "report" else "Восстановить сессию")
        command.add_argument("session_id")
        command.add_argument("--db", type=Path, default=Path("data/paper.sqlite3"))
        if name == "report":
            command.add_argument("--events", action="store_true", help="Включить полный журнал")
        else:
            command.add_argument("--interval-seconds", type=int, default=10)
    return cli


def execute(args):
    if args.command == "dashboard":
        from .dashboard import serve
        if not 1 <= args.port <= 65535:
            raise ValueError("Порт должен быть от 1 до 65535")
        serve(args.db, args.port, open_browser=args.open)
        return 0
    if args.command == "compare":
        from .evaluation import compare_session
        with writer_lock(args.db.resolve()):
            store = Store(args.db)
            try:
                config,state = store.load(args.session_id)
                if state.status != "CLOSED":
                    raise ValueError("Сначала завершите сессию")
                report = compare_session(dict(config=config.to_dict(),state=state.to_dict(),events=store.events(args.session_id)))
                store.save_analysis(args.session_id,"comparison",report)
                if args.output:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(encode(report),encoding="utf-8")
                print(encode(report))
                return 0
            finally:
                store.close()
    if args.command == "experiment":
        return run_experiment(args)
    if hasattr(args, "interval_seconds") and args.interval_seconds < 10:
        raise ValueError("Минимальный интервал публичных запросов — 10 секунд")
    if args.command == "report":
        if not args.db.is_file():
            raise ValueError("Файл базы не найден")
        store = Store(args.db)
        try:
            config, _ = store.load(args.session_id)
            if config.source == "alpaca-paper":
                from .broker import BrokerEngine
                engine = BrokerEngine(store, args.session_id)
            else:
                engine = Engine(store, args.session_id)
            print(encode(engine.report()))
            if args.events:
                print(encode(store.events(args.session_id)))
        finally:
            store.close()
        return 0
    with writer_lock(args.db.resolve()):
        store = Store(args.db)
        try:
            if args.command == "resume":
                if store.analysis(args.session_id,"experiment_role"):
                    raise ValueError("Исследовательский прогон нельзя продолжить как обычную торговую сессию")
                if store.load(args.session_id)[0].source == "alpaca-paper":
                    raise ValueError("Сессии Alpaca восстанавливаются через панель после подключения Paper-счёта")
                engine = Engine(store, args.session_id)
            else:
                source = "synthetic" if args.command == "demo" else "kraken-public"
                config = settings(args, source)
                minutes = decimal(args.duration_minutes)
                if not 1 < minutes <= 1440:
                    raise ValueError("Длительность должна быть больше 1 и не больше 1440 минут")
                start = utc_now()
                engine = Engine.start(store, config, start, start + timedelta(minutes=float(minutes)))
            print(f"SESSION_ID={engine.state.session_id}", flush=True)
            print(f"SOURCE={engine.config.source}; DEADLINE_UTC={engine.state.deadline_at}", flush=True)
            if engine.state.status != "CLOSED":
                if engine.config.source == "synthetic":
                    print("DEMO: искусственные цены, ускоренное модельное время; это не бэктест.")
                    run_demo(engine)
                else:
                    run_public(engine, args.interval_seconds)
            print(encode(engine.report()))
            return 2 if engine.state.status == "UNRESOLVED" else 0
        finally:
            store.close()


def run_experiment(args):
    from .experiment import Experiment
    from .qwen import OllamaClient
    from .history import CandleFeed, load_kraken_candles
    from .runtime import QuoteFeed
    from .watchdog import Watchdog
    minutes = decimal(args.duration_minutes)
    if not 1 < minutes <= 1440:
        raise ValueError("Длительность: больше 1 и до 1440 минут")
    client = OllamaClient()
    info = client.status()
    config = Config(source="kraken-public",capital=decimal(args.capital),
                    strategy_version="qwen-local-v1",llm_digest=info["digest"])
    quotes, candles = QuoteFeed(public_quote), CandleFeed(load_kraken_candles,"kraken-public")
    wake, trip = threading.Event(), []
    def on_trip(reason):
        trip.append(reason)
        wake.set()
    watchdog = Watchdog(on_trip)
    experiment = None
    with writer_lock(args.db.resolve()):
        store = Store(args.db)
        try:
            now = utc_now()
            experiment = Experiment.start(store,config,now,now+timedelta(minutes=float(minutes)),client=client)
            experiment.engines["qwen"].strategy.wake = wake
            print("FORWARD: общие публичные данные; три локальных виртуальных портфеля, без внешних ордеров.",flush=True)
            quotes.refresh(); candles.refresh(now); watchdog.start()
            while True:
                wake.clear()
                now = utc_now()
                quote,error = quotes.take()
                quotes.refresh(); candles.refresh(now)
                experiment.step(now,quote,error=error,stop=bool(trip),candle_context=candles.view(now))
                watchdog.progress()
                report = experiment.report()
                statuses = [r["status"] for r in report["results"].values()]
                if all(s in {"CLOSED","UNRESOLVED"} for s in statuses):
                    break
                wake.wait(1)
        except KeyboardInterrupt:
            if experiment:
                experiment.step(utc_now(),None,stop=True)
        finally:
            quotes.close(); candles.close(); watchdog.stop()
            if experiment:
                report = experiment.persist_report()
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(encode(report),encoding="utf-8")
                experiment.close()
            store.close()
    print(encode(report))
    return 2 if any(r["status"] == "UNRESOLVED" for r in report["results"].values()) else 0


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        return execute(parser().parse_args())
    except (ValueError, sqlite3.Error, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
