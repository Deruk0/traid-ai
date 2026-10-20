"""Measured quote analytics and bounded factual memory. No forecast or execution."""
from copy import deepcopy
from datetime import timedelta
import hashlib

from .models import D, encode, parse_time, stamp
from .strategy import DemoMomentum

VERSION = "quote-analytics-v1"
MAX_SAMPLES = 600
RETENTION_SECONDS = 3600
MEMORY_EVENTS = 8
HORIZONS = (60, 300, 900)


def number(value):
    return str(value.quantize(D("0.000001")))


def remember_quote(state, quote, now):
    """Called only after Engine validates source, freshness and ordering."""
    cutoff = parse_time(quote.observed_at) - timedelta(seconds=RETENTION_SECONDS)
    samples = [s for s in state.quote_samples if parse_time(s["observed_at"]) >= cutoff]
    samples.append(dict(quote_id=quote.quote_id, source=quote.source, instrument=quote.instrument,
                        observed_at=quote.observed_at, available_at=stamp(now),
                        bid=str(quote.bid), ask=str(quote.ask), mid=str(quote.mid)))
    state.quote_samples = samples[-MAX_SAMPLES:]


def window(samples, end, seconds, max_gap):
    boundary = end - timedelta(seconds=seconds)
    earlier = [s for s in samples if parse_time(s["observed_at"]) <= boundary]
    rows = ([earlier[-1]] if earlier else []) + [
        s for s in samples if parse_time(s["observed_at"]) > boundary]
    times = [parse_time(s["observed_at"]) for s in rows]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    complete = bool(earlier and len(rows) > 1 and
                    (boundary - times[0]).total_seconds() <= max_gap)
    gap_count = sum(g > max_gap for g in gaps)
    ready = complete and not gap_count
    prices = [D(s["mid"]) for s in rows]
    change = (prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else None
    returns = [(b / a - 1) * 100 for a, b in zip(prices, prices[1:])]
    # Sampled variation only: not candle volatility, annualized risk or a forecast.
    variation = sum((r * r for r in returns), D("0")).sqrt() if ready else None
    return dict(horizon_seconds=seconds, status="READY" if ready else "GAPS" if gap_count else "WARMUP",
                sample_count=len(rows), start_at=stamp(times[0]) if times else None,
                end_at=stamp(end), observed_span_seconds=(end-times[0]).total_seconds() if times else 0,
                gap_count=gap_count, largest_gap_seconds=max(gaps, default=0),
                return_pct=number(change) if ready else None,
                observed_change_pct=number(change) if change is not None else None,
                range_pct=number((max(prices)-min(prices))/prices[0]*100) if prices else None,
                sampled_variation_pct=number(variation) if variation is not None else None)


def analyze(config, state, quote, now):
    end = parse_time(quote.observed_at)
    cutoff = end - timedelta(seconds=RETENTION_SECONDS)
    rows = [s for s in state.quote_samples
            if s["source"] == quote.source and s["instrument"] == quote.instrument
            and cutoff <= parse_time(s["observed_at"]) <= end
            and parse_time(s["available_at"]) <= now]
    # build_snapshot is also callable directly; include its validated current quote.
    if not rows or rows[-1]["quote_id"] != quote.quote_id:
        rows.append(dict(quote_id=quote.quote_id, source=quote.source, instrument=quote.instrument,
                         observed_at=quote.observed_at, available_at=stamp(now),
                         bid=str(quote.bid), ask=str(quote.ask), mid=str(quote.mid)))
    windows = [window(rows, end, h, config.max_quote_age) for h in HORIZONS]
    ready = [w for w in windows if w["status"] == "READY"]
    positive = any(D(w["return_pct"]) > 0 for w in ready)
    negative = any(D(w["return_pct"]) < 0 for w in ready)
    signal = DemoMomentum().decide(state.history, quote.mid, D(state.quantity) > 0)
    spread_bps = (quote.ask-quote.bid)/quote.mid*10000
    round_trip = config.fee_rate*2*10000 + config.slippage_bps*2 + spread_bps
    return dict(version=VERSION, sample_count=len(rows), windows=windows,
                provenance=dict(source=quote.source, instrument=quote.instrument,
                                start_at=rows[0]["observed_at"], end_at=quote.observed_at,
                                available_at=stamp(now),
                                sample_set_sha256=hashlib.sha256(encode(rows).encode()).hexdigest()),
                quality=dict(data_kind="sampled_bid_ask", candle_history_available=False,
                             volume_available=False, exchange_event_time_available=quote.source != "kraken-public",
                             conflicting_horizons=positive and negative,
                             gap_threshold_seconds=config.max_quote_age),
                costs=dict(estimated_round_trip_bps=number(round_trip),
                           basis="configured_fees_and_slippage_plus_current_spread", is_estimate=True),
                reference_signal=dict(strategy_version=DemoMomentum.version, action=signal.action,
                                      reason_code=signal.reason, validated_trading_edge=False))


def changes_since_request(state, quote, now):
    previous = state.context_memory.get("last_request")
    if (not previous or previous["source"] != quote.source
            or parse_time(previous["observed_at"]) > parse_time(quote.observed_at)
            or parse_time(previous["available_at"]) > now):
        return None
    return dict(previous_snapshot_id=previous["snapshot_id"],
                elapsed_seconds=(parse_time(quote.observed_at)-parse_time(previous["observed_at"])).total_seconds(),
                mid_change_pct=number((quote.mid/D(previous["mid"])-1)*100),
                spread_change_bps=number((quote.ask-quote.bid)/quote.mid*10000-D(previous["spread_bps"])))


def session_memory(state, now):
    memory = state.context_memory
    recent = [e for e in memory.get("recent_events", []) if
              parse_time(state.started_at) <= parse_time(e["occurred_at"]) <= now]
    decision = memory.get("last_decision")
    if decision and parse_time(decision["occurred_at"]) > now:
        decision = None
    return dict(entries=state.entries, exits=state.exits, realized_pnl=state.realized_pnl,
                last_decision=deepcopy(decision), recent_events=deepcopy(recent[-MEMORY_EVENTS:]))


def decision_options(allowed, analytics):
    """Alternatives to review, not pre-approved signals or new strategies."""
    signal = analytics["reference_signal"]
    ready = [w["horizon_seconds"] for w in analytics["windows"] if w["status"] == "READY"]
    options = [dict(action="WAIT", reason_code="NO_PROVEN_EDGE_OR_INSUFFICIENT_EVIDENCE")]
    for action in allowed:
        if action != "WAIT":
            options.append(dict(action=action, reference_rule_supports=signal["action"] == action,
                                ready_horizons_seconds=ready, requires_independent_risk_check=True))
    return options


def context_summary(analytics, memory):
    return dict(sample_count=analytics["sample_count"],
                ready_horizons_seconds=[w["horizon_seconds"] for w in analytics["windows"] if w["status"] == "READY"],
                gaps=any(w["status"] == "GAPS" for w in analytics["windows"]),
                memory_event_count=len(memory["recent_events"]), volume_available=False)


def remember_events(state, now, events):
    """Project only allowlisted facts; never copy account IDs, errors or LLM prose."""
    memory = deepcopy(state.context_memory)
    recent = memory.get("recent_events", [])
    for kind, payload in events:
        if kind == "AI_REQUEST":
            snapshot = payload["snapshot"]
            memory["last_request"] = dict(snapshot_id=snapshot["snapshot_id"], source=snapshot["source"],
                observed_at=snapshot["observed_at"], available_at=now, mid=snapshot["market"]["mid"],
                spread_bps=snapshot["market"]["spread_bps"])
        elif kind == "AI_DECISION":
            memory["last_decision"] = dict(occurred_at=now, action=payload["action"],
                                           accepted=payload["accepted"], snapshot_id=payload["snapshot_id"])
        elif kind in {"PAPER_FILL", "BROKER_FILL"}:
            recent.append(dict(occurred_at=now, kind=kind, side=payload["side"],
                               quantity=payload["quantity"], price=payload["price"]))
        elif kind == "RISK_VERDICT" and not payload["approved"]:
            recent.append(dict(occurred_at=now, kind=kind, approved=False,
                               reason_code=payload["reason"]))
        elif kind == "SESSION_STATUS":
            recent.append(dict(occurred_at=now, kind=kind, status=payload["to"]))
        elif kind in {"DATA_UNAVAILABLE", "DATA_REJECTED", "ORDER_UNKNOWN", "ORDER_REJECTED",
                      "ORDER_ABORTED", "BROKER_UNRESOLVED"}:
            fact = dict(occurred_at=now, kind=kind)
            if recent and recent[-1]["kind"] == kind:
                recent[-1] = fact
            else:
                recent.append(fact)
    memory["recent_events"] = recent[-MEMORY_EVENTS:]
    state.context_memory = memory
