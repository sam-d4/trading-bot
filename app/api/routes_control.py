import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.broker.oanda_client import OandaClientError
from app.core.events import Event, EventType, event_bus
from app.persistence import repo
from app.persistence.db import get_session
from app.persistence.models import ModelStatus

router = APIRouter()


@router.post("/kill-switch/reset")
def reset_kill_switch(request: Request, session: Session = Depends(get_session)):
    """Never called automatically - only ever hit from an explicit dashboard action, so a human
    consciously reviews what happened before trading resumes."""
    client = request.app.state.oanda_client
    risk_manager = request.app.state.risk_manager
    if client is None:
        raise HTTPException(400, "OANDA client not configured - no credentials set")

    try:
        summary = client.get_account_summary()
    except OandaClientError as exc:
        raise HTTPException(502, f"failed to fetch account summary: {exc}") from exc

    now = dt.datetime.now(dt.timezone.utc)
    risk_manager.reset(summary.nav, now)
    repo.record_kill_switch_event(session, event="reset", reason="manual_resume", nav_at_event=summary.nav)
    event_bus.publish(Event(EventType.KILL_SWITCH_RESET, {"nav": summary.nav}))
    return {"status": "resumed", "nav": summary.nav}


@router.post("/kill-switch/trip")
def emergency_stop(request: Request, session: Session = Depends(get_session)):
    """The dashboard's Emergency Stop button - halts trading immediately via the same
    halted-state mechanism the automatic drawdown kill-switch uses (RiskManager.manual_trip),
    then always flattens open positions regardless of the flatten_on_kill_switch setting - an
    emergency stop that leaves positions open wouldn't actually be one. Resuming afterwards
    still requires the normal explicit 'Resume trading' action; this endpoint never auto-resumes."""
    client = request.app.state.oanda_client
    risk_manager = request.app.state.risk_manager
    order_manager = request.app.state.order_manager
    if client is None or order_manager is None:
        raise HTTPException(400, "OANDA client not configured - no credentials set")

    try:
        summary = client.get_account_summary()
    except OandaClientError as exc:
        raise HTTPException(502, f"failed to fetch account summary: {exc}") from exc

    now = dt.datetime.now(dt.timezone.utc)
    check = risk_manager.manual_trip(summary.nav, now, reason="manual emergency stop")
    repo.record_kill_switch_event(session, event="tripped", reason=check.reason or "manual emergency stop", nav_at_event=summary.nav)
    order_manager.flatten_all(session)
    event_bus.publish(Event(EventType.KILL_SWITCH_TRIPPED, {"reason": check.reason}))
    return {"status": "stopped", "nav": summary.nav}


@router.post("/models/{model_id}/promote")
def promote_model(model_id: int, session: Session = Depends(get_session)):
    """Flips a VALIDATED candidate to LIVE - the manual-confirmation step in the self-improvement
    loop's promotion gate. Only meaningful once app/rl/registry.py (Phase 7/8) exists to put
    models into VALIDATED status; wired in now so the dashboard/API contract is stable."""
    mv = repo.set_model_status(session, model_id, ModelStatus.LIVE)
    if mv is None:
        raise HTTPException(404, "model not found")
    event_bus.publish(Event(EventType.MODEL_PROMOTED, {"model_id": model_id, "name": mv.name}))
    return {"status": "promoted", "model_id": model_id}


@router.post("/models/{model_id}/reject")
def reject_model(model_id: int, session: Session = Depends(get_session)):
    mv = repo.set_model_status(session, model_id, ModelStatus.REJECTED)
    if mv is None:
        raise HTTPException(404, "model not found")
    return {"status": "rejected", "model_id": model_id}
