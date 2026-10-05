"""Translates a strategy's directional signal into an OANDA order, after the risk manager has
sized and cleared it. Refuses all new orders while the kill-switch is halted - this check is
deliberately duplicated here (the engine also shouldn't call in while halted) as defense in depth,
since the risk manager's halted state is what actually protects real money.
"""

import datetime as dt

import structlog
from sqlalchemy.orm import Session

from app.broker.models import OrderResult
from app.broker.oanda_client import OandaClient, OandaClientError
from app.core.events import Event, EventType, event_bus
from app.execution.reconciliation import _reconcile_locally_open_but_remotely_closed
from app.persistence import repo
from app.persistence.models import TradeDirection, TradeStatus
from app.risk.manager import RiskManager
from app.risk.position_sizing import size_position
from app.risk.limits import RiskLimits

log = structlog.get_logger(__name__)


class OrderManager:
    def __init__(self, client: OandaClient, risk_manager: RiskManager, limits: RiskLimits):
        self._client = client
        self._risk_manager = risk_manager
        self._limits = limits

    def execute_signal(
        self,
        session: Session,
        *,
        instrument: str,
        direction: int,  # +1 long, -1 short, 0 flat
        price: float,
        atr: float,
        equity: float,
        strategy_name: str,
        model_version_id: int | None = None,
        account_to_quote_rate: float = 1.0,
    ) -> OrderResult | None:
        if self._risk_manager.halted:
            log.warning("order_refused_kill_switch_halted", instrument=instrument, direction=direction)
            return None

        existing = repo.open_trades(session)
        existing_for_instrument = next((t for t in existing if t.instrument == instrument), None)

        if existing_for_instrument is not None:
            currently_long = existing_for_instrument.direction == TradeDirection.LONG
            wants_long = direction > 0
            if direction == 0 or currently_long != wants_long:
                if not self._close_local_trade(session, existing_for_instrument, reason="signal_reversed_or_flat"):
                    # Still open at OANDA - opening the opposite side now would net against it.
                    log.error("not_opening_new_trade_old_one_still_open", instrument=instrument)
                    return None
            else:
                # already positioned the direction the strategy wants - no pyramiding in v1.
                return None

        if direction == 0:
            return None

        sizing = size_position(
            equity=equity,
            price=price,
            atr=atr,
            direction=direction,
            limits=self._limits,
            account_to_quote_rate=account_to_quote_rate,
        )
        if sizing.units == 0:
            return None

        try:
            result = self._client.place_market_order(instrument, sizing.units, stop_distance=sizing.stop_distance)
        except OandaClientError as exc:
            log.error("order_placement_failed", instrument=instrument, units=sizing.units, error=str(exc))
            return None

        if result.status != "filled" or not result.trade_id:
            log.warning("order_not_filled", instrument=instrument, status=result.status)
            return result

        trade_direction = TradeDirection.LONG if direction > 0 else TradeDirection.SHORT
        trade = repo.record_trade_opened(
            session,
            oanda_trade_id=result.trade_id,
            instrument=instrument,
            direction=trade_direction,
            units=sizing.units,
            entry_price=result.fill_price or price,
            opened_at=dt.datetime.now(dt.timezone.utc),
            strategy_name=strategy_name,
            model_version_id=model_version_id,
        )
        event_bus.publish(
            Event(
                EventType.TRADE_OPENED,
                {
                    "instrument": trade.instrument,
                    "direction": trade.direction.value,
                    "units": trade.units,
                    "entry_price": trade.entry_price,
                    "strategy_name": trade.strategy_name,
                },
            )
        )
        return result

    def flatten_all(self, session: Session) -> None:
        """Called on kill-switch trip when flatten_on_kill_switch is enabled."""
        for trade in repo.open_trades(session):
            self._close_local_trade(session, trade, reason="kill_switch_flatten")

    def _close_local_trade(self, session: Session, trade, *, reason: str) -> bool:
        """True if the trade is now closed (by us, or already by OANDA's own stop); False if it is
        still open at OANDA and the caller must not act as though it were flat."""
        try:
            result = self._client.close_trade(trade.oanda_trade_id)
        except OandaClientError as exc:
            log.error("trade_close_failed", trade_id=trade.oanda_trade_id, error=str(exc), reason=reason)
            # With server-side stops this is often just "OANDA already stopped it out and our
            # local row hasn't caught up yet" - resolve that now instead of waiting for the next
            # periodic reconciliation pass.
            _reconcile_locally_open_but_remotely_closed(session, self._client, trade)
            session.refresh(trade)
            return trade.status != TradeStatus.OPEN

        if result.fill_price is None:
            # No fill transaction came back, so OANDA did NOT close it. Recording a close here left
            # a 1.9M-unit position open at OANDA while the dashboard showed none; leave the row
            # open and let the next signal / reconciliation pass retry.
            log.error("trade_close_not_filled", trade_id=trade.oanda_trade_id, reason=reason, raw=result.raw)
            return False

        now = dt.datetime.now(dt.timezone.utc)
        exit_price = result.fill_price
        direction_sign = 1 if trade.direction == TradeDirection.LONG else -1
        # OANDA's own realised P/L is in the ACCOUNT currency. The local (exit-entry)*units is in
        # the pair's QUOTE currency (JPY for USD_JPY, USD for AUD_USD), so on a GBP account it was
        # recorded as if it were GBP - a 1,133 JPY loss showed as a 7,153 "loss" on the dashboard.
        fill = (result.raw or {}).get("orderFillTransaction", {})
        oanda_pl = fill.get("pl")
        realized_pnl = (
            float(oanda_pl)
            if oanda_pl is not None
            else direction_sign * (exit_price - trade.entry_price) * trade.units
        )

        repo.record_trade_closed(
            session,
            oanda_trade_id=trade.oanda_trade_id,
            exit_price=exit_price,
            closed_at=now,
            realized_pnl=realized_pnl,
        )
        event_bus.publish(
            Event(
                EventType.TRADE_CLOSED,
                {
                    "instrument": trade.instrument,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "reason": reason,
                },
            )
        )
        return True
