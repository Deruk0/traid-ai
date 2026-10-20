"""Pure, revisionable reconciliation of confirmed Paper fills and posted fees."""
from datetime import date, timedelta
import hashlib

from .models import D, decimal, encode, parse_time, stamp


def _symbol(value):
    return str(value or "").replace("/", "").upper()


def _activity_time(activity):
    value = activity.get("transaction_time") or activity.get("date")
    if not isinstance(value, str):
        raise ValueError("Missing activity time")
    if len(value) == 10:
        return date.fromisoformat(value), None
    instant = parse_time(value)
    return instant.date(), instant


def reconcile_activities(config, state, events, activities, now, *, account, positions,
                         open_orders, other_order_ids=()):
    """Rebuild each time; never add posted fees to previous estimated accounting.

    Asset fees are signed costs: debits positive, USD rebates negative. A final
    report means reconciled as of checked_at, and can change on a later read.
    """
    now = parse_time(stamp(now))
    started = parse_time(state.started_at)
    unresolved, pending, unallocated = set(), set(), set()
    owned, journal, fill_days = {}, {}, set()
    other_order_ids = set(other_order_ids)
    fill_count = 0
    cash, quantity = config.capital, D(0)
    fee_usd, usd_cost, btc_cost = D(0), D(0), D(0)
    fee_days, posted_fee_count = set(), 0
    if not isinstance(events, list):
        events = []
        unresolved.add("MALFORMED_JOURNAL")

    # Broker journal deltas are authoritative; ORDER_UPDATE only proves identity.
    try:
        if config.source != "alpaca-paper" or not state.broker:
            raise ValueError("Not Paper accounting")
        initial_cash = decimal(state.broker["initial_cash"])
        for event in events:
            kind, payload = event["kind"], event["payload"]
            if kind not in {"BROKER_FILL", "ORDER_UPDATE"}:
                continue
            order_id = payload.get("order_id")
            side = str(payload.get("side", "")).lower()
            if not isinstance(order_id, str) or not order_id or side not in {"buy", "sell"}:
                raise ValueError("Invalid journal identity")
            if order_id in owned and owned[order_id] != side:
                raise ValueError("Conflicting journal side")
            owned[order_id] = side
            if kind == "BROKER_FILL":
                instant = parse_time(event["occurred_at"])
                if instant < started or instant > now:
                    raise ValueError("Invalid journal time")
                qty, price = decimal(payload["quantity"]), decimal(payload["price"])
                if qty <= 0 or price <= 0:
                    raise ValueError("Invalid journal fill")
                total = journal.setdefault(order_id, dict(qty=D(0), value=D(0), days=set()))
                total["qty"] += qty
                total["value"] += qty*price
                total["days"].add(instant.date())
                fill_days.add(instant.date())
                fill_count += 1
                cash += qty*price*(1 if side == "sell" else -1)
                quantity += qty*(1 if side == "buy" else -1)
            elif decimal(payload.get("filled_qty", "0")) < 0:
                raise ValueError("Invalid journal quantity")
        if set(owned) & other_order_ids:
            unresolved.add("ORDER_OWNERSHIP_CONFLICT")
    except (KeyError, TypeError, ValueError, AttributeError):
        initial_cash = None
        unresolved.add("MALFORMED_JOURNAL")

    unique = {}
    if not isinstance(activities, list):
        activities = []
        unresolved.add("MALFORMED_ACTIVITIES")
    for activity in activities:
        if not isinstance(activity, dict) or not isinstance(activity.get("id"), str) or not activity["id"]:
            unresolved.add("MALFORMED_ACTIVITY")
            continue
        activity_id = activity["id"]
        if activity_id in unique and encode(unique[activity_id]) != encode(activity):
            unresolved.add("CONFLICTING_ACTIVITY_ID")
            unallocated.add(activity_id)
        else:
            unique[activity_id] = activity
    activities_hash = hashlib.sha256(encode([unique[key] for key in sorted(unique)]).encode()).hexdigest()

    fills, dated_fills, fee_records = {}, [], []
    for activity_id, activity in unique.items():
        try:
            day, instant = _activity_time(activity)
            if day > now.date() or (instant is not None and instant > now):
                raise ValueError("Future activity")
            kind = activity.get("activity_type")
            order_id = activity.get("order_id")
            if kind == "FILL":
                if instant is None:
                    raise ValueError("Missing fill timestamp")
                dated_fills.append((day, order_id))
                if order_id not in owned or order_id in other_order_ids:
                    if instant >= started:
                        unresolved.add("EXTERNAL_FILL")
                        unallocated.add(activity_id)
                    continue
                side = str(activity.get("side", "")).lower()
                qty, price = decimal(activity["qty"]), decimal(activity["price"])
                if (_symbol(activity.get("symbol")) != "BTCUSD" or side != owned[order_id]
                        or qty <= 0 or price <= 0 or instant < started or order_id not in journal):
                    raise ValueError("Fill not represented by journal")
                total = fills.setdefault(order_id, dict(qty=D(0), value=D(0), days=set()))
                total["qty"] += qty
                total["value"] += qty*price
                total["days"].add(day)
            elif kind in {"CFEE", "FEE"}:
                fee_records.append((activity_id, activity, day))
            elif (instant >= started if instant is not None else day >= started.date()):
                unresolved.add("EXTERNAL_ACTIVITY")
                unallocated.add(activity_id)
        except (ValueError, KeyError, TypeError, AttributeError):
            unresolved.add("MALFORMED_ACTIVITY")
            unallocated.add(activity_id)

    raw_fills_match = not (set(fills)-set(journal))
    for order_id, expected in journal.items():
        actual = fills.get(order_id)
        if actual is None or actual["qty"] < expected["qty"]-config.quantity_step:
            pending.add("MISSING_FILL_ACTIVITIES")
            raw_fills_match = False
        elif (abs(actual["qty"]-expected["qty"]) > config.quantity_step
              or abs(actual["value"]-expected["value"]) > D(".01")):
            unresolved.add("FILL_MISMATCH")
            raw_fills_match = False
    # ORDER_UPDATE with a filled quantity but no BROKER_FILL cannot be explained.
    last_order_quantities = {}
    for event in events:
        try:
            if event["kind"] == "ORDER_UPDATE":
                payload = event["payload"]
                last_order_quantities[payload["order_id"]] = decimal(payload.get("filled_qty", "0"))
        except (TypeError, KeyError, ValueError):
            unresolved.add("MALFORMED_JOURNAL")
            raw_fills_match = False
    for order_id, filled_qty in last_order_quantities.items():
        if filled_qty > 0 and (order_id not in journal or
                abs(filled_qty-journal[order_id]["qty"]) > config.quantity_step):
            unresolved.add("FILL_WITHOUT_JOURNAL")
            raw_fills_match = False

    # Activity timestamps, rather than local receipt days, anchor EOD fee attribution.
    traded_days = set().union(*(f["days"] for f in fills.values())) if fills else fill_days
    for activity_id, activity, day in fee_records:
        try:
            order_id = activity.get("order_id")
            if order_id:
                if order_id not in journal or order_id in other_order_ids:
                    unresolved.add("UNALLOCATED_FEE")
                    unallocated.add(activity_id)
                    continue
                candidate_days = journal[order_id]["days"] | fills.get(order_id, {}).get("days", set())
            else:
                candidates = {day, day-timedelta(days=1)}
                candidate_days = traded_days & candidates
                foreign = any(fill_day in candidates and (oid not in owned or oid in other_order_ids)
                              for fill_day, oid in dated_fills)
                if not candidate_days or foreign:
                    unresolved.add("UNALLOCATED_FEE")
                    unallocated.add(activity_id)
                    continue
            if activity.get("status", "executed") != "executed":
                pending.add("FEE_NOT_POSTED")
                continue
            net = decimal(activity["net_amount"])
            if activity["activity_type"] == "CFEE":
                qty, price = decimal(activity["qty"]), decimal(activity["price"])
                if (_symbol(activity.get("symbol")) not in {"BTC", "BTCUSD"}
                        or net != 0 or qty > 0 or price <= 0):
                    raise ValueError("Invalid crypto fee valuation")
                quantity += qty
                btc_cost -= qty
                fee_usd -= qty*price
                nonzero = qty != 0
            else:
                if (decimal(activity.get("qty", "0")) != 0
                        or _symbol(activity.get("symbol")) not in {"", "USD", "BTCUSD"}
                        or activity.get("currency", "USD") != "USD"):
                    raise ValueError("Invalid USD fee")
                cash += net
                usd_cost -= net
                fee_usd -= net
                nonzero = net != 0
            if nonzero:
                fee_days.update(candidate_days)
                posted_fee_count += 1
        except (ValueError, KeyError, TypeError, AttributeError):
            unresolved.add("MALFORMED_FEE")
            unallocated.add(activity_id)

    has_posted_fees = bool(posted_fee_count and fee_usd > 0 and traded_days <= fee_days)
    if journal and not has_posted_fees:
        pending.add("MISSING_FEE_ACTIVITIES")
    if any(day >= now.date() for day in traded_days):
        pending.add("END_OF_DAY_FEES_PENDING")
    if state.status != "CLOSED":
        pending.add("SESSION_NOT_CLOSED")

    cash_match = qty_match = False
    actual_cash = actual_qty = None
    try:
        if (not isinstance(account, dict) or account.get("account_id") != state.broker["account_id"]
                or account.get("currency") != "USD"):
            raise ValueError("Invalid account identity/currency")
        actual_cash = decimal(account["cash"])
        if not isinstance(positions, list) or not isinstance(open_orders, list):
            raise ValueError("Invalid account positions/orders")
        actual_qty = D(0)
        for position in positions:
            if _symbol(position.get("symbol")) != "BTCUSD":
                unresolved.add("EXTERNAL_POSITION")
                continue
            qty = decimal(position["qty"])
            if qty < 0:
                raise ValueError("Negative spot position")
            actual_qty += qty
        if open_orders:
            unresolved.add("OPEN_ORDERS")
        if state.broker.get("pending") and not state.broker["pending"].get("terminal"):
            unresolved.add("PENDING_ORDER")
        if initial_cash is not None:
            cash_match = abs(actual_cash-(initial_cash+cash-config.capital)) <= D(".01")
            qty_match = abs(actual_qty-quantity) <= config.quantity_step*max(1, fill_count)
            if not cash_match:
                unresolved.add("CASH_MISMATCH")
            if not qty_match or quantity < 0:
                unresolved.add("POSITION_MISMATCH")
    except (ValueError, KeyError, TypeError, AttributeError):
        unresolved.add("MALFORMED_ACCOUNT")

    flat = quantity == 0 and actual_qty == 0
    if state.status == "CLOSED" and not flat:
        unresolved.add("RESIDUAL_POSITION")
    fills_match = bool(raw_fills_match and not unresolved)
    balances_match = bool(fills_match and cash_match and qty_match)
    complete = bool(not unresolved and not pending and fills_match and balances_match and flat)
    if not journal and complete:
        pending.add("NO_TRADES")  # informational, not an incomplete state
    status = "RECONCILED_AS_OF" if complete else "UNRESOLVED" if unresolved else "PENDING_ACTIVITIES"
    return dict(status=status, checked_at=stamp(now), fees_usd=str(fee_usd),
                fees_by_asset={"USD": str(usd_cost), "BTC": str(btc_cost)},
                net_pnl=str(cash-config.capital) if complete else None,
                cash=str(cash), quantity=str(quantity), actual_cash=str(actual_cash) if actual_cash is not None else None,
                actual_quantity=str(actual_qty) if actual_qty is not None else None,
                balances_match=balances_match, fills_match=fills_match, has_posted_fees=has_posted_fees,
                unallocated_activity_ids=sorted(unallocated), reason_codes=sorted(unresolved | pending),
                activity_ids=sorted(unique), activities_sha256=activities_hash, pnl_is_final=complete,
                fee_is_estimated=not complete, fees_are_estimated=not complete)
