"""Soft session target: accounting context only; never grants trading authority."""
from .models import D


def session_objective(config, state):
    if config.target_return is None:
        return None
    qty, quote = D(state.quantity), state.quote()
    net_equity = D(state.cash) if qty == 0 else None
    if qty > 0 and quote:
        net_equity = D(state.cash)+qty*quote.bid*(1-config.slippage_bps/10000)*(1-config.fee_rate)
    target = config.capital*config.target_return
    pnl = net_equity-config.capital if net_equity is not None else None
    progress = min(D(100),max(D(0),pnl/target*100)) if pnl is not None and target > 0 else None
    return dict(kind="SOFT_SESSION_TARGET", target_percent=str(config.target_return*100),
        target_pnl_usd=str(target),target_equity_usd=str(config.capital+target),
        estimated_net_pnl_usd=str(pnl) if pnl is not None else None,
        remaining_usd=str(max(D(0),target-pnl)) if pnl is not None else None,
        progress_percent=str(progress) if progress is not None else None,
        reached=pnl >= target if pnl is not None else None,
        valuation_observed_at=quote.observed_at if quote else None,
        includes_estimated_exit_costs=qty > 0, mandatory=False,
        may_raise_risk=False, may_extend_deadline=False)
