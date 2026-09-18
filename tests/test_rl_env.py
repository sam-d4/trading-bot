import numpy as np
import pandas as pd
import pytest

from app.data.features import feature_columns_for
from app.rl.env import TradingEnv

LOOKBACK = 3


def _features_df(log_returns: list[float]) -> pd.DataFrame:
    """A minimal features_df with a controlled log_return sequence - the feature columns
    themselves are irrelevant to reward math, only log_return drives it."""
    n = len(log_returns)
    data = {"time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")}
    for col in ["open", "high", "low", "close", "volume", *feature_columns_for("EUR_USD")]:
        data[col] = np.zeros(n)
    data["log_return"] = log_returns
    return pd.DataFrame(data)


def test_pnl_mode_reward_equals_return_minus_transaction_cost():
    returns = [0.0] * (LOOKBACK + 1) + [0.01, -0.02, 0.03]
    cost_bps = 1.0
    env = TradingEnv(
        _features_df(returns), lookback=LOOKBACK, reward_mode="pnl", flat_penalty_bps=0.0, transaction_cost_bps=cost_bps
    )
    env.reset()

    _, reward, _, _, _ = env.step(1)  # flat -> long: one full unit of transaction cost
    assert reward == pytest.approx(0.01 - cost_bps / 10_000)

    _, reward, _, _, _ = env.step(1)  # long -> long: no new transaction cost
    assert reward == pytest.approx(-0.02)


def test_flat_penalty_only_applies_when_position_is_flat():
    returns = [0.0] * (LOOKBACK + 2)
    env = TradingEnv(_features_df(returns), lookback=LOOKBACK, reward_mode="pnl", flat_penalty_bps=5.0)
    env.reset()

    _, reward_flat, _, _, _ = env.step(0)  # stay flat
    assert reward_flat == pytest.approx(-5.0 / 10_000)

    env.reset()
    _, reward_long, _, _, _ = env.step(1)  # go long - no flat penalty, just transaction cost
    assert reward_long != pytest.approx(-5.0 / 10_000)


def test_differential_sharpe_first_step_falls_back_to_raw_return():
    # with zero prior variance history, the DSR formula's denominator is ~0 - the implementation
    # must fall back to plain net_return rather than dividing by (near-)zero.
    returns = [0.0] * (LOOKBACK + 1) + [0.02]
    env = TradingEnv(
        _features_df(returns), lookback=LOOKBACK, reward_mode="differential_sharpe", flat_penalty_bps=0.0
    )
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert reward == pytest.approx(info["net_return"])


def test_differential_sharpe_matches_hand_computed_reference():
    """Reproduces Moody & Saffell's differential Sharpe ratio update by hand for a short,
    fully-deterministic return sequence and checks the env's output matches exactly."""
    eta = 0.1
    net_returns = [0.01, 0.01, -0.03, 0.02]  # transaction cost/flat penalty zeroed out - stays long throughout anyway
    returns = [0.0] * (LOOKBACK + 1) + net_returns
    env = TradingEnv(
        _features_df(returns),
        lookback=LOOKBACK,
        reward_mode="differential_sharpe",
        flat_penalty_bps=0.0,
        transaction_cost_bps=0.0,
        dsr_eta=eta,
    )
    env.reset()

    A, B = 0.0, 0.0
    expected = []
    for r in net_returns:
        dA, dB = r - A, r**2 - B
        var = B - A**2
        d = r if var <= 1e-8 else (B * dA - 0.5 * A * dB) / var**1.5
        expected.append(d)
        A += eta * dA
        B += eta * dB

    actual = []
    for _ in net_returns:
        _, reward, _, _, _ = env.step(1)  # stay long throughout - action is irrelevant to the reward math itself
        actual.append(reward)

    for e, a in zip(expected, actual, strict=True):
        assert a == pytest.approx(e, abs=1e-9)


def test_differential_sharpe_penalizes_volatility_relative_to_pnl():
    """The core promise of DSR: given two return sequences with the same average return, the
    lower-volatility one should score a higher cumulative DSR reward - a distinction raw "pnl"
    mode cannot make (its cumulative reward is just the average return, identical for both)."""
    n_tail = 40
    rng = np.random.default_rng(0)
    steady = np.full(n_tail, 0.001)
    volatile = 0.001 + rng.normal(0, 0.02, n_tail)
    volatile -= volatile.mean() - 0.001  # force identical mean return to isolate the volatility effect

    def cumulative_reward(tail: np.ndarray, mode: str) -> float:
        returns = [0.0] * (LOOKBACK + 1) + list(tail)
        env = TradingEnv(
            _features_df(returns), lookback=LOOKBACK, reward_mode=mode, flat_penalty_bps=0.0, transaction_cost_bps=0.0
        )
        env.reset()
        total = 0.0
        for _ in tail:
            _, reward, _, _, _ = env.step(1)
            total += reward
        return total

    steady_pnl = cumulative_reward(steady, "pnl")
    volatile_pnl = cumulative_reward(volatile, "pnl")
    assert steady_pnl == pytest.approx(volatile_pnl, abs=1e-6)  # pnl mode is blind to volatility, as expected

    steady_dsr = cumulative_reward(steady, "differential_sharpe")
    volatile_dsr = cumulative_reward(volatile, "differential_sharpe")
    assert steady_dsr > volatile_dsr  # DSR mode rewards the steadier path more, for the same P&L
