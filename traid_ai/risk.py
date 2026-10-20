from decimal import Decimal, ROUND_DOWN
from .models import Config, D


def entry_quantity(config: Config, cash: Decimal, equity: Decimal,
                   fill_price: Decimal, stop_price: Decimal) -> Decimal:
    """One position, immediate simulated fills: no outstanding order reservations."""
    if min(cash, equity, fill_price, stop_price) <= 0 or stop_price >= fill_price:
        return D("0")
    # Include estimated fees at both ends and extra adverse exit slippage.
    exit_price = stop_price * (1 - config.slippage_bps / 10000)
    unit_loss = fill_price - exit_price + config.fee_rate * (fill_price + exit_price)
    risk_qty = config.capital * config.risk_per_trade / unit_loss
    budget_qty = min(cash, equity * config.max_exposure) / (fill_price * (1 + config.fee_rate))
    qty = (min(risk_qty, budget_qty) / config.quantity_step).to_integral_value(rounding=ROUND_DOWN)
    qty *= config.quantity_step
    return qty if qty * fill_price >= config.min_notional else D("0")
