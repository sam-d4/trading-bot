import datetime as dt
from pathlib import Path

import pandas as pd
import structlog

from app.config.settings import Settings, get_settings

log = structlog.get_logger(__name__)


def historical_dir() -> Path:
    d = get_settings().data_dir / "historical"
    d.mkdir(parents=True, exist_ok=True)
    return d


def candle_path(instrument: str, granularity: str) -> Path:
    return historical_dir() / f"{instrument}_{granularity}.parquet"


def save_candles(df: pd.DataFrame, instrument: str, granularity: str) -> Path:
    path = candle_path(instrument, granularity)
    df.to_parquet(path, index=False)
    return path


def load_candles(instrument: str, granularity: str) -> pd.DataFrame:
    path = candle_path(instrument, granularity)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached candles at {path}. Run scripts/fetch_historical_data.py first."
        )
    return pd.read_parquet(path)


def refresh_cached_candles(settings: Settings, instrument: str, granularity: str = "H1") -> pd.DataFrame:
    """Brings the on-disk candle cache up to now by fetching only what's missing.

    Nothing used to do this: the cache is written by scripts/fetch_historical_data.py and was never
    updated again, so the retraining pipeline trained AND validated every candidate on data that
    ended weeks earlier (the last Sep 11 when this was found on Oct 5) - the same stale
    holdout window, reused by every cycle, blind to the market the bot was actually trading."""
    from app.data.candles import fetch_candles  # local import: keeps this module importable without oandapyV20

    cached = load_candles(instrument, granularity)
    last = cached["time"].max().to_pydatetime()
    now = dt.datetime.now(dt.timezone.utc)
    new = fetch_candles(settings, instrument, granularity, last + dt.timedelta(seconds=1), now)
    if new.empty:
        return cached
    merged = pd.concat([cached, new], ignore_index=True).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    save_candles(merged, instrument, granularity)
    log.info("candle_cache_refreshed", instrument=instrument, added=len(merged) - len(cached), ends=str(merged["time"].max()))
    return merged
