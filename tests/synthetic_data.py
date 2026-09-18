"""Shared synthetic OHLCV generator for RL pipeline tests - produces a primary instrument's
candles plus its three related instruments (see app/data/features.py's related_instruments),
since FEATURE_COLUMNS now always includes cross-pair columns and every real caller is expected
to supply related_candles. Tests that skip this and call compute_features(df) alone get a
KeyError building TradingEnv's observation matrix - which is exactly the shape-mismatch this
convention exists to catch, so don't "fix" it by making cross-pair features optional again."""

import numpy as np
import pandas as pd

from app.data.features import compute_features, related_instruments


def _synthetic_candles(n: int, *, seed: int, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 0.015, n)
    prices = 1.10 + drift + np.cumsum(rng.normal(0, 0.0004, n))
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="h", tz="UTC"),
            "open": prices,
            "high": prices + rng.random(n) * 0.0004,
            "low": prices - rng.random(n) * 0.0004,
            "close": prices + rng.normal(0, 0.0001, n),
            "volume": rng.integers(1, 500, n),
        }
    )


def synthetic_features(instrument: str = "EUR_USD", n: int = 800, seed: int = 3) -> pd.DataFrame:
    primary = _synthetic_candles(n, seed=seed)
    related = {
        inst: _synthetic_candles(n, seed=seed + i + 1) for i, inst in enumerate(related_instruments(instrument))
    }
    return compute_features(primary, related_candles=related, carry_differential=0.0)
