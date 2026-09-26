"""Historical candle fetch from OANDA, paginated into the ~5000-candle-per-request limit.

Used to build the multi-year dataset that the baseline strategies are backtested against and
that the RL policy is pretrained/fine-tuned on (see the plan's "cold-start knowledge" section) -
this is deliberately separate from the live OANDA client since it's a bulk, offline concern.
"""

import datetime as dt

import pandas as pd
import structlog
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

from app.config.settings import Settings

log = structlog.get_logger(__name__)

MAX_CANDLES_PER_REQUEST = 5000


class CandleFetchError(RuntimeError):
    pass


def _parse_candles(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for c in raw:
        if not c.get("complete", True):
            continue
        mid = c["mid"]
        rows.append(
            {
                "time": pd.Timestamp(c["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
            }
        )
    return pd.DataFrame(rows)


def fetch_candles(
    settings: Settings,
    instrument: str,
    granularity: str,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    """Fetch and concatenate all candles for [start, end), walking forward in
    MAX_CANDLES_PER_REQUEST-sized windows since OANDA caps a single request's count."""
    environment = "practice" if settings.oanda_environment == "practice" else "live"
    api = API(access_token=settings.oanda_api_token, environment=environment, request_params={"timeout": 30})

    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        # OANDA rejects a request that sets 'count' together with both 'from' and 'to' - so
        # page with 'from' + 'count' only, and trim anything past `end` after the loop instead.
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": MAX_CANDLES_PER_REQUEST,
        }
        req = InstrumentsCandles(instrument=instrument, params=params)
        try:
            resp = api.request(req)
        except V20Error as exc:
            raise CandleFetchError(f"failed fetching candles for {instrument}: {exc}") from exc

        batch = _parse_candles(resp.get("candles", []))
        if batch.empty:
            break
        frames.append(batch)

        last_time = batch["time"].iloc[-1].to_pydatetime()
        if last_time <= cursor:
            break  # safety valve against an infinite loop if OANDA returns no forward progress
        cursor = last_time + dt.timedelta(seconds=1)

        log.info(
            "candle_batch_fetched",
            instrument=instrument,
            granularity=granularity,
            batch_rows=len(batch),
            cursor=cursor.isoformat(),
        )

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time").sort_values("time")
    result = result[result["time"] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)
