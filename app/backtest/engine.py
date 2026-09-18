"""Bar-by-bar backtest replay.

Reused for two purposes: (1) validating the baseline strategy library in Phase 2/3, and (2) the
RL validation gate in Phase 7/8, which backtests each retrained candidate the same way before
it's eligible for promotion - one engine, one definition of "how do we measure edge."
"""

from dataclasses import dataclass

import pandas as pd

from app.backtest.metrics import summarize
from app.strategy.base import Signal, Strategy

GRANULARITY_PERIODS_PER_YEAR = {
    "M1": 365 * 24 * 60,
    "M5": 365 * 24 * 12,
    "M15": 365 * 24 * 4,
    "H1": 365 * 24,
    "H4": 365 * 6,
    "D": 365,
}


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # indexed by `time`, starts at 1.0
    returns: pd.Series  # per-bar strategy return (after transaction costs)
    trade_returns: pd.Series  # return realized on each closed round-trip
    metrics: dict


def run_backtest(
    features_df: pd.DataFrame,
    strategy: Strategy,
    *,
    granularity: str = "H1",
    transaction_cost_bps: float = 1.0,
) -> BacktestResult:
    """features_df: output of app.data.features.compute_features (must include `time`,
    `log_return`, and the feature columns the strategy reads). transaction_cost_bps: round-turn
    cost charged in basis points of notional whenever the position changes."""
    df = features_df.reset_index(drop=True)
    signals = df.apply(strategy.decide, axis=1)
    position = signals.map({Signal.FLAT: 0, Signal.LONG: 1, Signal.SHORT: -1})

    # A signal decided using bar t's features can only be acted on for bar t+1's return -
    # using it against bar t's own return would be lookahead bias.
    applied_position = position.shift(1).fillna(0)

    gross_return = applied_position * df["log_return"]

    position_changed = applied_position.diff().fillna(applied_position.iloc[0]).abs()
    cost = position_changed * (transaction_cost_bps / 10_000)
    net_return = gross_return - cost

    equity_curve = (1 + net_return).cumprod()
    equity_curve.index = df["time"]

    # Extract per-round-trip returns: a "trade" closes whenever position flips or goes flat.
    trade_returns = _extract_trade_returns(applied_position, net_return)

    periods_per_year = GRANULARITY_PERIODS_PER_YEAR.get(granularity, 365 * 24)
    metrics = summarize(net_return, equity_curve, trade_returns, periods_per_year)

    return BacktestResult(
        equity_curve=equity_curve, returns=net_return, trade_returns=trade_returns, metrics=metrics
    )


def _extract_trade_returns(position: pd.Series, net_return: pd.Series) -> pd.Series:
    trade_returns = []
    running = 0.0
    in_trade = False
    for pos, ret in zip(position, net_return, strict=True):
        if pos != 0:
            running += ret
            in_trade = True
        elif in_trade:
            trade_returns.append(running)
            running = 0.0
            in_trade = False
    if in_trade:
        trade_returns.append(running)
    return pd.Series(trade_returns)


def buy_and_hold_result(features_df: pd.DataFrame, granularity: str = "H1") -> BacktestResult:
    df = features_df.reset_index(drop=True)
    net_return = df["log_return"]
    equity_curve = (1 + net_return).cumprod()
    equity_curve.index = df["time"]
    periods_per_year = GRANULARITY_PERIODS_PER_YEAR.get(granularity, 365 * 24)
    metrics = summarize(net_return, equity_curve, pd.Series([equity_curve.iloc[-1] - 1.0]), periods_per_year)
    return BacktestResult(equity_curve=equity_curve, returns=net_return, trade_returns=pd.Series([]), metrics=metrics)
