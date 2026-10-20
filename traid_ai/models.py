from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import re


D = Decimal
LLM_MODELS = ("qwen3.5:4b", "qwen3.5:9b")


def decimal(value: object) -> Decimal:
    try:
        result = D(str(value))
    except InvalidOperation as exc:
        raise ValueError("Некорректное число") from exc
    if not result.is_finite():
        raise ValueError("Число должно быть конечным")
    return result


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Нужен часовой пояс")
    return value.astimezone(timezone.utc).isoformat()


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Время должно содержать часовой пояс")
    return result.astimezone(timezone.utc)


def encode(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class Config:
    source: str
    capital: Decimal = D("1000")
    risk_per_trade: Decimal = D("0.005")
    max_exposure: Decimal = D("0.25")
    loss_limit: Decimal = D("0.03")
    drawdown_limit: Decimal = D("0.05")
    stop_fraction: Decimal = D("0.02")
    fee_rate: Decimal = D("0.001")
    slippage_bps: Decimal = D("5")
    quantity_step: Decimal = D("0.00000001")
    min_notional: Decimal = D("5")
    max_quote_age: int = 30
    max_hold_seconds: int | None = None
    flatten_buffer_seconds: int = 60
    strategy_version: str = "demo-momentum-v1"
    schema_version: int = 1
    llm_digest: str = ""
    decision_policy: str = "review-v1"
    exit_policy: str = "trailing-stop-v1"
    target_return: Decimal | None = None
    llm_thinking: bool = True
    llm_model: str = "qwen3.5:4b"

    def __post_init__(self):
        if self.source not in {"synthetic", "kraken-public", "alpaca-paper"}:
            raise ValueError("Неизвестный источник сессии")
        for name in ("capital", "risk_per_trade", "max_exposure", "loss_limit",
                     "drawdown_limit", "stop_fraction", "fee_rate", "slippage_bps",
                     "quantity_step", "min_notional"):
            value = getattr(self, name)
            if not isinstance(value, D) or not value.is_finite():
                raise ValueError(f"{name}: требуется конечный Decimal")
        if self.capital <= 0 or self.quantity_step <= 0 or self.min_notional <= 0:
            raise ValueError("Бюджет, шаг количества и минимум ордера должны быть > 0")
        for name in ("risk_per_trade", "max_exposure", "loss_limit",
                     "drawdown_limit", "stop_fraction"):
            if not D("0") < getattr(self, name) <= D("1"):
                raise ValueError(f"{name} должен быть в диапазоне (0, 1]")
        if not 0 <= self.fee_rate < 1 or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Некорректные расходы исполнения")
        if min(self.max_quote_age, self.flatten_buffer_seconds) <= 0:
            raise ValueError("Интервалы должны быть положительными")
        if self.max_hold_seconds is not None and (type(self.max_hold_seconds) is not int or self.max_hold_seconds <= 0):
            raise ValueError("Максимум удержания: положительное целое число секунд либо не задан")
        if self.strategy_version not in {"demo-momentum-v1", "paper-smoke-v1", "qwen-local-v1"} or self.schema_version != 1:
            raise ValueError("Версия конфигурации/стратегии не поддерживается")
        if self.strategy_version == "paper-smoke-v1" and self.source != "alpaca-paper":
            raise ValueError("Проверка ордеров доступна только в Alpaca Paper")
        if self.strategy_version == "qwen-local-v1" and self.source == "synthetic":
            raise ValueError("Qwen работает с сессиями в реальном времени; демо ускорено")
        if self.llm_digest and not re.fullmatch(r"[a-f0-9]{64}", self.llm_digest):
            raise ValueError("Некорректная версия локальной модели")
        if not isinstance(self.llm_thinking, bool):
            raise ValueError("llm_thinking должен быть bool")
        if self.llm_model not in LLM_MODELS:
            raise ValueError("Выберите Qwen 3.5 4B или 9B")
        if self.decision_policy not in {"legacy", "evidence-v1", "review-v1"}:
            raise ValueError("Неизвестная версия правил решений")
        if self.exit_policy not in {"fixed-stop-v1", "trailing-stop-v1"}:
            raise ValueError("Неизвестное правило сопровождения позиции")
        if self.exit_policy == "trailing-stop-v1" and self.stop_fraction >= 1:
            raise ValueError("Расстояние скользящего стопа должно быть меньше 100 %")
        if self.target_return is not None and (not isinstance(self.target_return,D)
                or not self.target_return.is_finite() or self.target_return < 0):
            raise ValueError("Цель сессии: конечное неотрицательное число либо не задана")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Config":
        data = dict(payload)
        # Existing sessions keep their original strategy/answer contract on restore.
        data.setdefault("decision_policy", "legacy")
        data.setdefault("exit_policy", "fixed-stop-v1")
        data.setdefault("target_return", None)
        data.setdefault("llm_thinking", False)
        data.setdefault("max_hold_seconds", 3600)
        if data["target_return"] is not None:
            data["target_return"] = decimal(data["target_return"])
        for key in ("capital", "risk_per_trade", "max_exposure", "loss_limit",
                    "drawdown_limit", "stop_fraction", "fee_rate", "slippage_bps",
                    "quantity_step", "min_notional"):
            data[key] = decimal(data[key])
        return cls(**data)


@dataclass(frozen=True)
class Quote:
    quote_id: str
    observed_at: str
    bid: Decimal
    ask: Decimal
    source: str
    instrument: str = "BTC/USD"

    def validate(self, now: datetime, config: Config) -> None:
        if self.source != config.source or self.instrument != "BTC/USD":
            raise ValueError("Источник или инструмент не соответствует сессии")
        if not self.quote_id or not self.bid.is_finite() or not self.ask.is_finite():
            raise ValueError("Некорректная котировка")
        if self.bid <= 0 or self.ask < self.bid:
            raise ValueError("Некорректный bid/ask")
        age = (now - parse_time(self.observed_at)).total_seconds()
        if age < 0 or age > config.max_quote_age:
            raise ValueError("Котировка из будущего или устарела")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2


@dataclass
class State:
    session_id: str
    started_at: str
    deadline_at: str
    status: str
    cash: str
    peak_equity: str
    quantity: str = "0"
    entry_price: str = "0"
    entry_cost: str = "0"
    stop_price: str = "0"
    entered_at: str | None = None
    last_quote: dict | None = None
    last_tick_at: str | None = None
    history: list[str] = field(default_factory=list)
    fees: str = "0"
    slippage_cost: str = "0"
    realized_pnl: str = "0"
    max_drawdown: str = "0"
    entries: int = 0
    exits: int = 0
    exit_reason: str | None = None
    broker: dict | None = None
    ai: dict | None = None
    quote_samples: list[dict] = field(default_factory=list)
    context_memory: dict = field(default_factory=dict)
    candle_context: dict = field(default_factory=dict)
    accounting: dict = field(default_factory=dict)
    position_peak: str = "0"
    position_exit_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def quote(self) -> Quote | None:
        if not self.last_quote:
            return None
        data = dict(self.last_quote)
        data["bid"], data["ask"] = D(data["bid"]), D(data["ask"])
        return Quote(**data)
