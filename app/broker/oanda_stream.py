"""Async wrapper around OANDA's pricing stream.

oandapyV20's PricingStream endpoint is a synchronous, blocking generator (built on `requests`'
streaming support), so it's run in a background thread and ticks are handed back to the asyncio
event loop via a thread-safe queue - this keeps the rest of the app (FastAPI, the trading loop)
fully async without needing a separate async HTTP streaming client.

NOT YET VERIFIED AGAINST A LIVE OANDA STREAM (no account credentials available while this was
written) - the message shape assumed below (type PRICE/HEARTBEAT, bids/asks arrays) matches
OANDA's documented v20 streaming format, but should be confirmed against real traffic in Phase 5
verification once credentials exist.
"""

import asyncio
import json
import threading

import structlog
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream as PricingStreamEndpoint

from app.broker.models import Tick
from app.config.settings import Settings

log = structlog.get_logger(__name__)


class OandaPriceStream:
    """instruments: one or more instruments to stream in a single connection - OANDA's
    PricingStream endpoint natively accepts a comma-separated instruments list, so tracking the
    primary instrument plus its cross-pair-feature related instruments (see
    app/data/features.py's related_instruments) costs one stream, not several. Each Tick carries
    its own `instrument` field so callers can route ticks per-instrument."""

    def __init__(self, settings: Settings, instruments: list[str]):
        self._settings = settings
        self._instruments = instruments
        environment = "practice" if settings.oanda_environment == "practice" else "live"
        # OANDA sends a heartbeat every ~5s, so a 30s read timeout means a silently-dead
        # connection raises (and gets reconnected) instead of hanging the thread forever.
        self._api = API(
            access_token=settings.oanda_api_token, environment=environment, request_params={"timeout": 30}
        )
        self._req: PricingStreamEndpoint | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def ticks(self) -> asyncio.Queue[Tick]:
        """Starts the background thread on first call and returns a queue of Tick objects.
        A None sentinel is never pushed; on stream failure a warning is logged and the thread
        exits - callers (core/engine.py) should treat a queue that stops producing as a
        disconnect and alert, per the plan's "stream disconnect" alert case."""
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=1000)

        self._thread = threading.Thread(target=self._run, args=(loop, queue), daemon=True)
        self._thread.start()
        return queue

    def stop(self) -> None:
        self._stop_event.set()
        if self._req is not None and hasattr(self._req, "terminate"):
            try:
                self._req.terminate("client stop requested")
            except Exception:  # pragma: no cover - best-effort, oandapyV20 internals not guaranteed
                pass

    def _new_request(self) -> PricingStreamEndpoint:
        return PricingStreamEndpoint(
            accountID=self._settings.oanda_account_id, params={"instruments": ",".join(self._instruments)}
        )

    def _run(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        """Reconnects with capped exponential backoff on any failure or clean stream end - a
        single network blip used to end this thread permanently, silently killing all live
        trading (the engine only alerts, it can't restart a thread that has exited)."""
        backoff = 1.0
        while not self._stop_event.is_set():
            self._req = self._new_request()
            try:
                for raw in self._api.request(self._req):
                    if self._stop_event.is_set():
                        return
                    backoff = 1.0
                    tick = _parse_tick(raw)
                    if tick is not None:
                        loop.call_soon_threadsafe(_put_nowait_drop_oldest, queue, tick)
                log.warning("price_stream_ended_reconnecting", instruments=self._instruments)
            except Exception as exc:  # noqa: BLE001 - any stream failure should reconnect, not kill the thread
                log.error("price_stream_error_reconnecting", instruments=self._instruments, error=str(exc), retry_in=backoff)
            if self._stop_event.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)


def _parse_tick(raw: dict | str) -> Tick | None:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if raw.get("type") != "PRICE":
        return None  # ignore HEARTBEAT messages
    bids = raw.get("bids", [])
    asks = raw.get("asks", [])
    if not bids or not asks:
        return None
    return Tick(
        instrument=raw["instrument"],
        time=raw["time"],
        bid=float(bids[0]["price"]),
        ask=float(asks[0]["price"]),
    )


def _put_nowait_drop_oldest(queue: asyncio.Queue, tick: Tick) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(tick)
