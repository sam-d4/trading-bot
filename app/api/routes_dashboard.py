import datetime as dt

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.persistence import repo
from app.persistence.db import get_session

router = APIRouter()


def _trade_dict(t) -> dict:
    return {
        "id": t.id,
        "instrument": t.instrument,
        "direction": t.direction.value,
        "units": t.units,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "opened_at": t.opened_at.isoformat(),
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        "realized_pnl": t.realized_pnl,
        "status": t.status.value,
        "strategy_name": t.strategy_name,
        "model_version_id": t.model_version_id,
    }


@router.get("/trades")
def list_trades(limit: int = Query(100, le=1000), session: Session = Depends(get_session)):
    return [_trade_dict(t) for t in repo.recent_trades(session, limit=limit)]


@router.get("/positions")
def list_open_positions(session: Session = Depends(get_session)):
    return [_trade_dict(t) for t in repo.open_trades(session)]


@router.get("/equity")
def equity_curve(hours: int = Query(24 * 7, le=24 * 365 * 5), session: Session = Depends(get_session)):
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    snaps = repo.equity_history(session, since=since)
    return [
        {"taken_at": s.taken_at.isoformat(), "nav": s.nav, "balance": s.balance, "unrealized_pnl": s.unrealized_pnl}
        for s in snaps
    ]


@router.get("/models")
def list_models(instrument: str | None = None, session: Session = Depends(get_session)):
    from sqlalchemy import select

    from app.persistence.models import ModelVersion

    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(200)
    if instrument is not None:
        stmt = stmt.where(ModelVersion.instrument == instrument)
    rows = session.scalars(stmt)
    return [
        {
            "id": m.id,
            "name": m.name,
            "kind": m.kind,
            "instrument": m.instrument,
            "status": m.status.value,
            "created_at": m.created_at.isoformat(),
            "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
            "validation_metrics": m.validation_metrics_json,
        }
        for m in rows
    ]


@router.get("/models/pending")
def pending_promotions(instrument: str | None = None, session: Session = Depends(get_session)):
    return [
        {"id": m.id, "name": m.name, "instrument": m.instrument, "validation_metrics": m.validation_metrics_json}
        for m in repo.pending_promotion_candidates(session, instrument)
    ]


@router.get("/status")
def status(request: Request, session: Session = Depends(get_session)):
    settings = request.app.state.settings
    risk_manager = request.app.state.risk_manager
    engine = getattr(request.app.state, "engine", None)
    engine_running = getattr(request.app.state, "engine_task", None) is not None

    live_by_instrument = repo.current_live_models_by_instrument(session, settings.traded_instruments)
    latest_snapshot = repo.latest_equity_snapshot(session)

    instruments = []
    for inst in settings.traded_instruments:
        live_model = live_by_instrument.get(inst)
        strategy_name = engine._strategies[inst].name if engine is not None and inst in engine._strategies else None
        instruments.append(
            {
                "instrument": inst,
                "strategy_name": strategy_name,
                "live_model": {"id": live_model.id, "name": live_model.name} if live_model else None,
            }
        )

    return {
        "environment": settings.oanda_environment,
        "engine_running": engine_running,
        "trading_halted": risk_manager.halted,
        "latest_nav": latest_snapshot.nav if latest_snapshot else None,
        "pending_promotions": len(repo.pending_promotion_candidates(session)),
        "instruments": instruments,
    }
