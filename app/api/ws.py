import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import event_bus

log = structlog.get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Relays every published Event to this dashboard client. No auth on the websocket itself
    in v1 (FastAPI's HTTPBasic dependency doesn't apply cleanly to the WS handshake) - acceptable
    for now since the dashboard is intended to sit behind the VPS firewall / an SSH tunnel, not
    be exposed directly; revisit if that changes."""
    await websocket.accept()
    queue = event_bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event.to_json())
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(queue)
