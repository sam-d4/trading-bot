import numpy as np
import pandas as pd

from app.backtest.engine import buy_and_hold_result, run_backtest
from app.data.features import compute_features
from app.strategy.base import Signal, Strategy


class AlwaysLongStrategy(Strategy):
    name = "always_long"

    def decide(self, features_row: pd.Series) -> Signal:
        return Signal.LONG


class AlwaysFlatStrategy(Strategy):
    name = "always_flat"

    def decide(self, features_row: pd.Series) -> Signal:
        return Signal.FLAT


def _synthetic_features(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices,
            "high": prices + rng.random(n) * 0.0004,
            "low": prices - rng.random(n) * 0.0004,
            "close": prices + rng.normal(0, 0.0001, n),
            "volume": rng.integers(1, 500, n),
        }
    )
    return compute_features(df)


def test_always_flat_has_no_trades_and_flat_equity():
    feats = _synthetic_features()
    result = run_backtest(feats, AlwaysFlatStrategy())
    assert result.metrics["trade_count"] == 0
    assert result.metrics["total_return_pct"] == 0.0


def test_always_long_tracks_buy_and_hold_direction():
    feats = _synthetic_features()
    long_result = run_backtest(feats, AlwaysLongStrategy(), transaction_cost_bps=0.0)
    bh_result = buy_and_hold_result(feats)
    # always-long with zero cost and a one-bar lag should be nearly identical to buy-and-hold
    assert abs(long_result.metrics["total_return_pct"] - bh_result.metrics["total_return_pct"]) < 1.0


def test_transaction_costs_reduce_returns():
    feats = _synthetic_features()
    no_cost = run_backtest(feats, AlwaysLongStrategy(), transaction_cost_bps=0.0)
    with_cost = run_backtest(feats, AlwaysLongStrategy(), transaction_cost_bps=50.0)
    assert with_cost.metrics["total_return_pct"] < no_cost.metrics["total_return_pct"]
