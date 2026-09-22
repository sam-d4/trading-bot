"""A wider library of well-known, publicly-documented FOREX trading strategies, added to
directly answer the question "would a strategy from a reliable published source do better than
what this project already tries?" rather than assuming the answer.

Sources (retail-TA definitions are standard enough to appear near-identically across references -
Investopedia is cited as a representative one; academic strategies cite the actual paper):
  - Moving average crossover, MACD, RSI, Bollinger Bands, Stochastic Oscillator, ADX/DMI,
    Parabolic SAR: standard technical-analysis definitions, e.g. Investopedia's entries on each.
  - Donchian channel breakout: Richard Donchian's original trend-following rule, popularized as
    the entry rule in the "Turtle Trading" system (Dennis & Eckhardt, 1983; see Michael Covel,
    "The Complete TurtleTrader", 2007).
  - Momentum + carry: Menkhoff, Sarno, Schmeling & Schrimpf, "Currency Momentum Strategies",
    Journal of Financial Economics (2012), combined with the classic FX carry trade (long
    higher-yield, short lower-yield) - two well-documented academic FX factors used together.

Each strategy reads pre-computed indicator columns from `add_extended_indicators()` below rather
than computing anything itself in `decide()`, matching this project's existing pattern (see
app/data/features.py / app/strategy/baselines.py): a Strategy's decide() is a stateless read of
one already-computed row, never a rolling calculation over history.

IMPORTANT SCOPE NOTE: `add_extended_indicators()` is deliberately NOT folded into
app/data/features.py's compute_features(), which is what the live engine and every RL model
actually run on. These strategies are for the one-off comparison in
scripts/backtest_strategy_library.py only. If one of them turns out to actually win the
validation gate and get promoted, its indicator columns need to be added to the live feature
pipeline as a separate, deliberate follow-up - not silently assumed to already be there.
"""

import numpy as np
import pandas as pd

from app.strategy.base import Signal, Strategy


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, smooth: int = 3):
    low_n = low.rolling(k_period).min()
    high_n = high.rolling(k_period).max()
    raw_k = 100 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(smooth).mean()
    return k, d


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Same formula as app/data/features.py's _atr - duplicated here because compute_features()
    only exposes atr_pct (atr/close) in its output, not the raw atr this needs."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Wilder's DMI/ADX."""
    atr = _wilder_atr(high, low, close, period=period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    plus_dm_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smoothed / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)


def _parabolic_sar(high: pd.Series, low: pd.Series, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Wilder's original Parabolic SAR (1978), the standard iterative definition - inherently
    path-dependent (each bar's value depends on the running trend/extreme-point state), so unlike
    every other indicator here this can't be a vectorized rolling formula."""
    high_v, low_v = high.to_numpy(), low.to_numpy()
    n = len(high_v)
    psar = np.zeros(n)
    bull = True
    af = af_step
    ep = low_v[0]
    psar[0] = low_v[0]
    for i in range(1, n):
        prev = psar[i - 1]
        prior_extreme = low_v[i - 2] if i >= 2 else low_v[i - 1]
        prior_extreme_high = high_v[i - 2] if i >= 2 else high_v[i - 1]
        if bull:
            candidate = prev + af * (ep - prev)
            candidate = min(candidate, low_v[i - 1], prior_extreme)
            if low_v[i] < candidate:
                bull, candidate, ep, af = False, ep, low_v[i], af_step
            elif high_v[i] > ep:
                ep, af = high_v[i], min(af + af_step, af_max)
        else:
            candidate = prev + af * (ep - prev)
            candidate = max(candidate, high_v[i - 1], prior_extreme_high)
            if high_v[i] > candidate:
                bull, candidate, ep, af = True, ep, high_v[i], af_step
            elif low_v[i] < ep:
                ep, af = low_v[i], min(af + af_step, af_max)
        psar[i] = candidate
    return pd.Series(psar, index=high.index)


def add_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: output of app.data.features.compute_features (has open/high/low/close/atr/rsi/
    momentum_return/carry_differential already). Returns a copy with the extra columns the
    strategies below need."""
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["donchian_high"] = high.rolling(20).max()
    df["donchian_low"] = low.rolling(20).min()

    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close)

    df["plus_di"], df["minus_di"], df["adx"] = _adx_dmi(high, low, close)

    df["psar"] = _parabolic_sar(high, low)

    return df


class MACrossoverStrategy(Strategy):
    """Classic "Golden Cross / Death Cross" trend-following rule: long while the fast (50) EMA
    is above the slow (200) EMA, short while below. Always in the market (no flat state) -
    faithful to the textbook definition rather than a variant."""

    name = "lib_ma_crossover"

    def decide(self, features_row: pd.Series) -> Signal:
        if pd.isna(features_row["ema50"]) or pd.isna(features_row["ema200"]):
            return Signal.FLAT
        return Signal.LONG if features_row["ema50"] > features_row["ema200"] else Signal.SHORT


class MACDTrendStrategy(Strategy):
    """Long while the MACD line is above its signal line, short while below - the standard
    MACD trend-state rule (vs. trading only the momentary crossover event)."""

    name = "lib_macd_trend"

    def decide(self, features_row: pd.Series) -> Signal:
        if pd.isna(features_row["macd"]) or pd.isna(features_row["macd_signal"]):
            return Signal.FLAT
        return Signal.LONG if features_row["macd"] > features_row["macd_signal"] else Signal.SHORT


class RSIReversalStrategy(Strategy):
    """Classic bounded-oscillator mean reversion: long when RSI(14) < 30 (oversold), short when
    RSI(14) > 70 (overbought), flat in between."""

    name = "lib_rsi_reversal"

    def decide(self, features_row: pd.Series) -> Signal:
        rsi = features_row["rsi"]
        if rsi < 30:
            return Signal.LONG
        if rsi > 70:
            return Signal.SHORT
        return Signal.FLAT


class BollingerMeanReversionStrategy(Strategy):
    """Fade the bands: long when price closes below the lower Bollinger band (20, 2 sigma),
    short when it closes above the upper band."""

    name = "lib_bollinger_mean_reversion"

    def decide(self, features_row: pd.Series) -> Signal:
        close, upper, lower = features_row["close"], features_row["bb_upper"], features_row["bb_lower"]
        if pd.isna(upper) or pd.isna(lower):
            return Signal.FLAT
        if close < lower:
            return Signal.LONG
        if close > upper:
            return Signal.SHORT
        return Signal.FLAT


class BollingerBreakoutStrategy(Strategy):
    """The opposite read of the same bands: trade WITH a breakout beyond them (momentum), not
    against it - long above the upper band, short below the lower band."""

    name = "lib_bollinger_breakout"

    def decide(self, features_row: pd.Series) -> Signal:
        close, upper, lower = features_row["close"], features_row["bb_upper"], features_row["bb_lower"]
        if pd.isna(upper) or pd.isna(lower):
            return Signal.FLAT
        if close > upper:
            return Signal.LONG
        if close < lower:
            return Signal.SHORT
        return Signal.FLAT


class DonchianBreakoutStrategy(Strategy):
    """Turtle-Trading-style 20-bar channel breakout: long on a new 20-bar high, short on a new
    20-bar low, flat otherwise (a simplified point-signal version of the original
    hold-until-opposite-channel-breached rule)."""

    name = "lib_donchian_breakout"

    def decide(self, features_row: pd.Series) -> Signal:
        close, hi, lo = features_row["close"], features_row["donchian_high"], features_row["donchian_low"]
        if pd.isna(hi) or pd.isna(lo):
            return Signal.FLAT
        if close >= hi:
            return Signal.LONG
        if close <= lo:
            return Signal.SHORT
        return Signal.FLAT


class StochasticReversalStrategy(Strategy):
    """Long when %K falls in the oversold zone (< 20), short when in the overbought zone
    (> 80) - the standard Stochastic(14,3,3) reversal-zone rule."""

    name = "lib_stochastic_reversal"

    def decide(self, features_row: pd.Series) -> Signal:
        k = features_row["stoch_k"]
        if pd.isna(k):
            return Signal.FLAT
        if k < 20:
            return Signal.LONG
        if k > 80:
            return Signal.SHORT
        return Signal.FLAT


class ADXTrendFilterStrategy(Strategy):
    """Only trade when ADX(14) > 25 signals a genuine trend (a standard ADX-as-filter rule),
    direction from which of +DI/-DI is dominant; flat when the market is judged non-trending."""

    name = "lib_adx_trend_filter"

    def decide(self, features_row: pd.Series) -> Signal:
        adx, plus_di, minus_di = features_row["adx"], features_row["plus_di"], features_row["minus_di"]
        if pd.isna(adx) or adx < 25:
            return Signal.FLAT
        return Signal.LONG if plus_di > minus_di else Signal.SHORT


class ParabolicSARStrategy(Strategy):
    """Always-in-market trend follower: long while price is above the Parabolic SAR dot, short
    while below - Wilder's original stop-and-reverse system used directly as an entry rule."""

    name = "lib_parabolic_sar"

    def decide(self, features_row: pd.Series) -> Signal:
        close, psar = features_row["close"], features_row["psar"]
        if pd.isna(psar):
            return Signal.FLAT
        return Signal.LONG if close > psar else Signal.SHORT


class MomentumCarryStrategy(Strategy):
    """Academic FX factor combination: only take a position when trailing momentum and the carry
    (interest-rate) differential AGREE on direction - long when both are positive, short when
    both are negative, flat when they disagree. Distinct from the project's existing
    baseline_carry (which ignores momentum entirely)."""

    name = "lib_momentum_carry"

    def decide(self, features_row: pd.Series) -> Signal:
        momentum, carry = features_row["momentum_return"], features_row["carry_differential"]
        if pd.isna(momentum):
            return Signal.FLAT
        if momentum > 0 and carry > 0:
            return Signal.LONG
        if momentum < 0 and carry < 0:
            return Signal.SHORT
        return Signal.FLAT


def strategy_library() -> list[Strategy]:
    return [
        MACrossoverStrategy(),
        MACDTrendStrategy(),
        RSIReversalStrategy(),
        BollingerMeanReversionStrategy(),
        BollingerBreakoutStrategy(),
        DonchianBreakoutStrategy(),
        StochasticReversalStrategy(),
        ADXTrendFilterStrategy(),
        ParabolicSARStrategy(),
        MomentumCarryStrategy(),
    ]
