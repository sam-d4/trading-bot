from pathlib import Path

import pandas as pd

from app.config.settings import get_settings


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
