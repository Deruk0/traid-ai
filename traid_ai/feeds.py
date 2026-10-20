from datetime import datetime
import json
import math
import urllib.error
import urllib.request
from uuid import uuid4

from .models import Config, D, Quote, decimal, stamp, utc_now


def synthetic_quote(index: int, now: datetime) -> Quote:
    # Artificial repeating path for testing; not historical BTC prices.
    price = D(f"{100 + 8 * math.sin(index / 4) + index * 0.08:.4f}")
    return Quote(f"synthetic-{index}", stamp(now), price - D("0.05"),
                 price + D("0.05"), "synthetic")


def parse_kraken(payload: dict, observed_at: datetime) -> Quote:
    if not isinstance(payload, dict) or payload.get("error") != []:
        raise ValueError("Источник котировок вернул ошибку")
    result = payload.get("result")
    if not isinstance(result, dict) or set(result) != {"XXBTZUSD"}:
        raise ValueError("Неожиданный инструмент в ответе источника")
    try:
        ticker = result["XXBTZUSD"]
        quote = Quote(str(uuid4()), stamp(observed_at), decimal(ticker["b"][0]),
                      decimal(ticker["a"][0]), "kraken-public")
        quote.validate(observed_at, Config(source="kraken-public"))
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Повреждённый ответ источника котировок") from exc
    return quote


def public_quote() -> Quote:
    """GET-only public ticker, hardcoded single symbol. No keys or trading API."""
    request = urllib.request.Request(
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        headers={"User-Agent": "TraidAI-Paper/0.1", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            data = response.read(65537)
        if len(data) > 65536:
            raise ValueError("Слишком большой ответ источника")
        return parse_kraken(json.loads(data), utc_now())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("Не удалось получить публичную котировку") from exc
