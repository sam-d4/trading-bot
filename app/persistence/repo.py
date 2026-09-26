"""Thin CRUD helpers over the SQLAlchemy models - callers (execution, risk, API) work with these
instead of writing ad-hoc queries, so persistence details stay in one place."""

import datetime as dt
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import (
    EquitySnapshot,
    KillSwitchEvent,
    ModelStatus,
    ModelVersion,
    Trade,
    TradeStatus,
)


def record_equity_snapshot(session: Session, *, nav: float, balance: float, unrealized_pnl: float) -> EquitySnapshot:
    snap = EquitySnapshot(nav=nav, balance=balance, unrealized_pnl=unrealized_pnl)
    session.add(snap)
    session.commit()
    session.refresh(snap)
    return snap


def latest_equity_snapshot(session: Session) -> EquitySnapshot | None:
    stmt = select(EquitySnapshot).order_by(EquitySnapshot.taken_at.desc()).limit(1)
    return session.scalars(stmt).first()


def equity_history(session: Session, *, since: dt.datetime | None = None, limit: int = 5000) -> list[EquitySnapshot]:
    stmt = select(EquitySnapshot).order_by(EquitySnapshot.taken_at.asc())
    if since is not None:
        stmt = stmt.where(EquitySnapshot.taken_at >= since)
    return list(session.scalars(stmt.limit(limit)))


def open_trade_by_oanda_id(session: Session, oanda_trade_id: str) -> Trade | None:
    stmt = select(Trade).where(Trade.oanda_trade_id == oanda_trade_id)
    return session.scalars(stmt).first()


def record_trade_opened(
    session: Session,
    *,
    oanda_trade_id: str,
    instrument: str,
    direction,
    units: float,
    entry_price: float,
    opened_at: dt.datetime,
    strategy_name: str,
    model_version_id: int | None = None,
) -> Trade:
    trade = Trade(
        oanda_trade_id=oanda_trade_id,
        instrument=instrument,
        direction=direction,
        units=abs(units),
        entry_price=entry_price,
        opened_at=opened_at,
        status=TradeStatus.OPEN,
        strategy_name=strategy_name,
        model_version_id=model_version_id,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def record_trade_closed(
    session: Session, *, oanda_trade_id: str, exit_price: float, closed_at: dt.datetime, realized_pnl: float
) -> Trade | None:
    trade = open_trade_by_oanda_id(session, oanda_trade_id)
    if trade is None:
        return None
    trade.exit_price = exit_price
    trade.closed_at = closed_at
    trade.realized_pnl = realized_pnl
    trade.status = TradeStatus.CLOSED
    session.commit()
    session.refresh(trade)
    return trade


def reopen_trade(session: Session, oanda_trade_id: str) -> Trade | None:
    """Undo a wrongly-recorded close: the local row says CLOSED but OANDA still has the trade open."""
    trade = session.scalars(select(Trade).where(Trade.oanda_trade_id == oanda_trade_id)).first()
    if trade is None:
        return None
    trade.status = TradeStatus.OPEN
    trade.exit_price = None
    trade.closed_at = None
    trade.realized_pnl = None
    session.commit()
    session.refresh(trade)
    return trade


def recent_trades(session: Session, *, limit: int = 200) -> list[Trade]:
    stmt = select(Trade).order_by(Trade.opened_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def open_trades(session: Session) -> list[Trade]:
    stmt = select(Trade).where(Trade.status == TradeStatus.OPEN)
    return list(session.scalars(stmt))


def record_kill_switch_event(session: Session, *, event: str, reason: str, nav_at_event: float) -> KillSwitchEvent:
    row = KillSwitchEvent(event=event, reason=reason, nav_at_event=nav_at_event)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def create_model_version(
    session: Session,
    *,
    name: str,
    kind: str,
    instrument: str,
    lookback: int,
    artifact_path: str | None = None,
    training_data_start: dt.datetime | None = None,
    training_data_end: dt.datetime | None = None,
) -> ModelVersion:
    mv = ModelVersion(
        name=name,
        kind=kind,
        instrument=instrument,
        lookback=lookback,
        artifact_path=artifact_path,
        training_data_start=training_data_start,
        training_data_end=training_data_end,
        status=ModelStatus.CANDIDATE,
    )
    session.add(mv)
    session.commit()
    session.refresh(mv)
    return mv


def _json_safe(value):
    """Recursively replaces non-finite floats (inf/-inf/nan) with None. Python's json.dumps
    happily emits the bare tokens Infinity/-Infinity/NaN by default (a non-standard JSON
    extension), which a spec-compliant JSON.parse - i.e. every browser - throws a SyntaxError on.
    A metrics dict with an infinite profit_factor (any strategy with zero losing trades, e.g.
    buy_and_hold, hits this constantly) would otherwise serialize "successfully" here and then
    fail to parse in the dashboard silently (frontend/src/components/TrainingPanel.tsx wraps its
    JSON.parse in a try/catch that swallows the error and just renders nothing)."""
    if isinstance(value, float) and not _is_finite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))  # x == x is False only for NaN


def set_model_status(
    session: Session, model_id: int, status: ModelStatus, *, validation_metrics: dict | None = None
) -> ModelVersion | None:
    mv = session.get(ModelVersion, model_id)
    if mv is None:
        return None
    mv.status = status
    if validation_metrics is not None:
        # allow_nan=False as a backstop: if _json_safe ever misses a non-finite value (e.g. a
        # new metrics shape added later), fail loudly here rather than writing invalid JSON that
        # only breaks, silently, much later in the dashboard.
        mv.validation_metrics_json = json.dumps(_json_safe(validation_metrics), allow_nan=False)
    if status == ModelStatus.LIVE:
        mv.promoted_at = dt.datetime.now(dt.timezone.utc)
    session.commit()
    session.refresh(mv)
    return mv


def current_live_model(session: Session, instrument: str | None = None) -> ModelVersion | None:
    """instrument=None returns the most-recently-promoted live model regardless of instrument -
    kept for the handful of callers (e.g. the dashboard's single-model status field) that
    predate multi-instrument support. Everything trading-decision-related should pass instrument
    explicitly (see current_live_models_by_instrument for fetching several at once)."""
    stmt = select(ModelVersion).where(ModelVersion.status == ModelStatus.LIVE)
    if instrument is not None:
        stmt = stmt.where(ModelVersion.instrument == instrument)
    stmt = stmt.order_by(ModelVersion.promoted_at.desc())
    return session.scalars(stmt).first()


def current_live_models_by_instrument(session: Session, instruments: list[str]) -> dict[str, ModelVersion]:
    """One DB round-trip for "what's live on each of these pairs right now" - used by the
    multi-instrument engine's model-hot-swap poll and the dashboard's per-pair status."""
    return {inst: mv for inst in instruments if (mv := current_live_model(session, inst)) is not None}


def pending_promotion_candidates(session: Session, instrument: str | None = None) -> list[ModelVersion]:
    stmt = select(ModelVersion).where(ModelVersion.status == ModelStatus.VALIDATED)
    if instrument is not None:
        stmt = stmt.where(ModelVersion.instrument == instrument)
    return list(session.scalars(stmt))
