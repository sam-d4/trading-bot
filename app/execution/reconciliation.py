"""Periodically reconciles local trade records against OANDA's actual account state, which is
always the source of truth - the local DB is a mirror for the dashboard/backtesting, not
authoritative. Catches cases order_manager.py can't see directly: a trade OANDA closed on its
own (stop-out, margin closeout) or a gap left by a crash/restart mid-order.
"""

import datetime as dt

import structlog
from sqlalchemy.orm import Session

from app.broker.oanda_client import OandaClient, OandaClientError
from app.core.events import Event, EventType, event_bus
from app.persistence import repo
from app.persistence.models import TradeDirection

log = structlog.get_logger(__name__)


def reconcile(session: Session, client: OandaClient) -> None:
    try:
        remote_open = client.get_open_trades()
    except OandaClientError as exc:
        log.error("reconciliation_fetch_failed", error=str(exc))
        return

    remote_ids = {t.trade_id for t in remote_open}
    local_open = repo.open_trades(session)

    for trade in local_open:
        if trade.oanda_trade_id in remote_ids:
            continue
        _reconcile_locally_open_but_remotely_closed(session, client, trade)

    local_ids = {t.oanda_trade_id for t in local_open}
    for remote_trade in remote_open:
        if remote_trade.trade_id in local_ids:
            continue
        # We DO have a row for it, but it says CLOSED while OANDA still has it open - a close that
        # was recorded locally without ever filling at OANDA. Trust OANDA and reopen the row.
        if repo.reopen_trade(session, remote_trade.trade_id) is not None:
            log.error("reopened_trade_wrongly_recorded_closed", trade_id=remote_trade.trade_id)
            continue
        # OANDA has an open trade we have no local record of (e.g. after a crash mid-order).
        # Adopt it so the dashboard reflects reality, tagged as untracked.
        log.warning("adopting_untracked_open_trade", trade_id=remote_trade.trade_id)
        direction = TradeDirection.LONG if remote_trade.units > 0 else TradeDirection.SHORT
        repo.record_trade_opened(
            session,
            oanda_trade_id=remote_trade.trade_id,
            instrument=remote_trade.instrument,
            direction=direction,
            units=abs(remote_trade.units),
            entry_price=remote_trade.price,
            opened_at=dt.datetime.now(dt.timezone.utc),
            strategy_name="untracked_adopted",
        )


def _reconcile_locally_open_but_remotely_closed(session: Session, client: OandaClient, trade) -> None:
    try:
        detail = client.get_trade(trade.oanda_trade_id)
    except OandaClientError as exc:
        log.error("reconciliation_trade_detail_failed", trade_id=trade.oanda_trade_id, error=str(exc))
        return

    if detail.get("state") != "CLOSED":
        return  # not actually closed remotely; leave it, will retry next reconciliation tick

    realized_pnl = float(detail.get("realizedPL", 0.0))
    # OANDA's TradeDetails doesn't reliably expose a single "close price" field across all
    # closure reasons (manual close vs. stop-out) - back it out from realized P&L instead,
    # which is always present and authoritative for accounting purposes.
    direction_sign = 1 if trade.direction == TradeDirection.LONG else -1
    exit_price = trade.entry_price + (realized_pnl / trade.units) * direction_sign if trade.units else trade.entry_price

    updated = repo.record_trade_closed(
        session,
        oanda_trade_id=trade.oanda_trade_id,
        exit_price=exit_price,
        closed_at=dt.datetime.now(dt.timezone.utc),
        realized_pnl=realized_pnl,
    )
    if updated is not None:
        event_bus.publish(
            Event(
                EventType.TRADE_CLOSED,
                {
                    "instrument": updated.instrument,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "reason": "reconciled_remote_close",
                },
            )
        )
