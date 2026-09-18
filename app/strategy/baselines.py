"""Rule-based baseline strategies.

These are not just a placeholder for Phase 5's live-loop test - per the plan's "cold-start
knowledge" approach, they are the bot's initial source of trading knowledge: the RL policy is
warm-started via behavior cloning against these strategies' decisions (see app/rl/pretrain.py)
rather than starting from randomly-initialized weights.

Each strategy reads a single row from app/data/features.py's output, so live trading and
backtesting see identical logic.
"""

import pandas as pd

from app.strategy.base import Signal, Strategy


class TrendFollowingStrategy(Strategy):
    """Long when the fast EMA is meaningfully above the slow EMA (and vice versa) - a classic
    trend-following/breakout rule. `momentum_trend` is (ema_fast - ema_slow) / close, so it's a
    scale-free measure of trend strength."""

    name = "baseline_trend_following"

    def __init__(self, trend_threshold: float = 0.003):
        # Calibrated against 5yr EUR_USD H1 backtest data (scripts/run_backtest.py): tighter
        # thresholds fire on noise and whipsaw (1390 trades/5yr at 0.0008, Sharpe -1.85). Wider
        # settings trade less and lose less, but never turn net-positive at any threshold tested
        # - this rule shows no measurable directional edge net of costs on that sample, which is
        # a normal result for a naive TA signal, not a bug. This default reduces noise-trading
        # for behavior-cloning purposes (app/rl/pretrain.py); it is not "the profitable setting".
        self.trend_threshold = trend_threshold

    def decide(self, features_row: pd.Series) -> Signal:
        trend = features_row["momentum_trend"]
        if trend > self.trend_threshold:
            return Signal.LONG
        if trend < -self.trend_threshold:
            return Signal.SHORT
        return Signal.FLAT


class MeanReversionStrategy(Strategy):
    """Fade extremes: long when price is unusually far below its rolling mean (oversold),
    short when unusually far above (overbought), measured via `meanrev_zscore`."""

    name = "baseline_mean_reversion"

    def __init__(self, entry_zscore: float = 3.5):
        # Same story as TrendFollowingStrategy above - widened from 1.5 to cut whipsaw trading
        # (2934 trades/5yr at 1.5, Sharpe -1.82); still net-negative at every zscore tested on
        # that sample, this just reduces noise for the behavior-cloning teacher signal.
        self.entry_zscore = entry_zscore

    def decide(self, features_row: pd.Series) -> Signal:
        z = features_row["meanrev_zscore"]
        if pd.isna(z):
            return Signal.FLAT
        if z < -self.entry_zscore:
            return Signal.LONG
        if z > self.entry_zscore:
            return Signal.SHORT
        return Signal.FLAT


class CarryStrategy(Strategy):
    """Long the higher-yielding currency, short the lower-yielding one, based on the interest
    rate differential. `carry_differential` defaults to 0 in the feature pipeline until it's
    wired up to a live rates feed (noted as a future enhancement) - until then this strategy is
    a documented no-op (always flat), not a strategy quietly pretending to have an edge it
    doesn't have data for."""

    name = "baseline_carry"

    def __init__(self, entry_differential: float = 0.5):
        self.entry_differential = entry_differential

    def decide(self, features_row: pd.Series) -> Signal:
        carry = features_row["carry_differential"]
        if carry > self.entry_differential:
            return Signal.LONG
        if carry < -self.entry_differential:
            return Signal.SHORT
        return Signal.FLAT


def default_baseline_strategies() -> list[Strategy]:
    return [TrendFollowingStrategy(), MeanReversionStrategy(), CarryStrategy()]
