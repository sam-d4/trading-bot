import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std(ddof=0) == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def win_rate(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize(
    returns: pd.Series,
    equity_curve: pd.Series,
    trade_returns: pd.Series,
    periods_per_year: float,
) -> dict:
    return {
        "total_return_pct": float((equity_curve.iloc[-1] - 1.0) * 100) if len(equity_curve) else 0.0,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown_pct": max_drawdown(equity_curve) * 100,
        "win_rate": win_rate(trade_returns),
        "profit_factor": profit_factor(trade_returns),
        "trade_count": int(len(trade_returns)),
    }
