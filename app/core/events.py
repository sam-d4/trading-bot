"""Minimal async pub/sub event bus - the live trading loop publishes typed events here, and
api/ws.py fans them out to connected dashboard clients. Kept in-process (no external broker like
Redis) since this is a single-instance, single-user system."""

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    EQUITY_UPDATE = "equity_update"
    KILL_SWITCH_TRIPPED = "kill_switch_tripped"
    KILL_SWITCH_RESET = "kill_switch_reset"
    TRAINING_STATUS = "training_status"
    MODEL_PROMOTION_PENDING = "model_promotion_pending"
    MODEL_PROMOTED = "model_promoted"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    def to_json(self) -> dict:
        return {"type": self.type.value, "payload": self.payload, "at": self.at.isoformat()}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, event: Event) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # a slow/disconnected dashboard client shouldn't block the trading loop
                pass


event_bus = EventBus()
