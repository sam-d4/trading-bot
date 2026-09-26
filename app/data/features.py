"""Quant-informed feature pipeline - the SAME functions are used for live trading and for
training/backtesting, so there is exactly one definition of "what the model sees" and no
train/serve skew.

Deliberately built on plain pandas/numpy rolling ops rather than the `pandas-ta` package: that
library's release cadence has fallen behind pandas itself (see the plan's noted staleness risk),
and it hung/broke against the pandas version this project installs. The indicators used here
(SMA/EMA, RSI, ATR, z-score) are simple enough to implement directly and are worth owning rather
than depending on a stale wrapper for.

Extended beyond the original price-only technical set with three richer input families, added
after the first round of RL training runs consistently found no learnable edge in plain
single-instrument technical indicators (see app/rl scheduler logs / scripts/run_training.py
history) - a well-known limitation of that feature class on liquid FX pairs:
  - cross-pair momentum (`related_candles`): other majors' recent momentum, since FX pairs move
    on shared USD-strength/risk-sentiment factors a single-pair view can't see.
  - multi-timeframe trend (`include_multi_timeframe`): H4/Daily trend context resampled from the
    same instrument's own bars - no extra data fetch needed, just a different lens on it.
  - carry (`carry_differential`): now wired to OANDA's real, live financing rates
    (app/broker/oanda_client.py's get_financing_rates) instead of a hardcoded 0. Note the
    historical-backtest caveat below - OANDA only exposes the CURRENT rate, not a rate history.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}
# Every column a compute_features() output DataFrame has besides these is a feature column -
# used to derive a DataFrame's actual feature list (app/rl/env.py's TradingEnv) or to compute
# what a given instrument's feature list WILL be without needing real data (feature_columns_for
# below, used wherever a caller needs the shape before any DataFrame exists yet, e.g. loading a
# saved model for live trading).
NON_FEATURE_COLUMNS = {"time", "open", "high", "low", "close", "volume", "log_return"}


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _trend_strength(close: pd.Series, window: int) -> pd.Series:
    ema_fast = _ema(close, span=max(window // 4, 2))
    ema_slow = _ema(close, span=window)
    return (ema_fast - ema_slow) / close


def _higher_timeframe_trend(df: pd.DataFrame, rule: str, offset: pd.Timedelta, window: int) -> pd.Series:
    """Resamples this instrument's own OHLC up to a coarser timeframe (H4/Daily) and computes a
    trend-strength reading on it, then shifts the bar's timestamp forward by its own duration so
    a merge_asof only ever sees a higher-timeframe bar that has actually finished closing as of
    the primary timestamp - the resampling equivalent of the one-bar-lag rule used everywhere
    else in this pipeline (see compute_features's `time`-indexed merge below)."""
    htf = df.set_index("time")["close"].resample(rule).last().dropna()
    trend = _trend_strength(htf, window=window)
    trend.index = trend.index + offset
    return trend


def _cross_pair_momentum(related: pd.DataFrame, window: int) -> pd.Series:
    series = related.sort_values("time").set_index("time")["close"].pct_change(window)
    return series


def compute_features(
    candles: pd.DataFrame,
    *,
    momentum_window: int = 20,
    meanrev_window: int = 20,
    atr_period: int = 14,
    rsi_period: int = 14,
    carry_differential: float = 0.0,
    related_candles: dict[str, pd.DataFrame] | None = None,
    include_multi_timeframe: bool = True,
) -> pd.DataFrame:
    """candles: DataFrame with columns time, open, high, low, close, volume, sorted ascending by time.

    carry_differential: the instrument's current long-vs-short financing rate spread (annualized),
    typically from OandaClient.get_financing_rates - see that function's docstring for the
    important historical-backtest caveat (OANDA only exposes the CURRENT rate, so a backtest
    necessarily treats it as constant across the whole window, not the true rate history).

    related_candles: other instruments' candle data (same shape as `candles`), keyed by
    instrument name (e.g. {"GBP_USD": df}) - adds each one's own recent momentum as a feature,
    since FX pairs move on shared USD-strength/risk-sentiment factors a single-pair view misses.
    Merged with merge_asof(direction="backward") so only already-known bars are ever used.

    include_multi_timeframe: adds H4 and Daily trend-strength context resampled from this same
    instrument's own bars (no extra data needed) - trend confirmation across timeframes is
    standard technical-analysis practice precisely because a single timeframe is noisy.
    """
    missing = REQUIRED_COLUMNS - set(candles.columns)
    if missing:
        raise ValueError(f"candles is missing required columns: {missing}")

    df = candles.sort_values("time").reset_index(drop=True).copy()

    log_return = np.log(df["close"] / df["close"].shift(1))

    # --- momentum: trailing return + trend strength (fast EMA vs slow EMA) ---
    df["momentum_return"] = df["close"].pct_change(momentum_window)
    df["momentum_trend"] = _trend_strength(df["close"], window=momentum_window)

    # --- mean-reversion: z-score of price vs its own rolling mean ---
    df["meanrev_zscore"] = _zscore(df["close"], meanrev_window)

    # --- volatility regime ---
    atr = _atr(df, period=atr_period)
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"]
    df["volatility_regime"] = _zscore(atr, window=atr_period * 4)

    # --- oscillator ---
    df["rsi"] = _rsi(df["close"], period=rsi_period)

    # --- session / time-of-day effects ---
    hour = df["time"].dt.hour
    df["session_sin"] = np.sin(2 * np.pi * hour / 24)
    df["session_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["time"].dt.dayofweek

    # --- carry (interest-rate differential) ---
    df["carry_differential"] = carry_differential

    # --- multi-timeframe trend context ---
    multi_tf_cols: list[str] = []
    if include_multi_timeframe:
        h4_trend = _higher_timeframe_trend(df, "4h", pd.Timedelta(hours=4), window=momentum_window)
        d_trend = _higher_timeframe_trend(df, "1D", pd.Timedelta(days=1), window=momentum_window)
        df = pd.merge_asof(df, h4_trend.rename("htf_trend_h4").reset_index(), on="time", direction="backward")
        df = pd.merge_asof(df, d_trend.rename("htf_trend_d").reset_index(), on="time", direction="backward")
        df["htf_trend_h4"] = df["htf_trend_h4"].fillna(0.0)
        df["htf_trend_d"] = df["htf_trend_d"].fillna(0.0)
        multi_tf_cols = ["htf_trend_h4", "htf_trend_d"]

    # --- cross-pair momentum context ---
    cross_pair_cols: list[str] = []
    if related_candles:
        for instrument, related_df in related_candles.items():
            col = f"xpair_{instrument.lower()}_momentum"
            momentum = _cross_pair_momentum(related_df, window=momentum_window)
            df = pd.merge_asof(df, momentum.rename(col).reset_index(), on="time", direction="backward")
            df[col] = df[col].fillna(0.0)
            cross_pair_cols.append(col)

    # --- raw return, useful as a reward/label input downstream ---
    df["log_return"] = log_return

    feature_cols = [
        "momentum_return",
        "momentum_trend",
        "meanrev_zscore",
        "atr_pct",
        "volatility_regime",
        "rsi",
        "session_sin",
        "session_cos",
        "day_of_week",
        "carry_differential",
        *multi_tf_cols,
        *cross_pair_cols,
    ]
    warmup = max(momentum_window, meanrev_window, atr_period * 4, rsi_period)
    df = df.iloc[warmup:].reset_index(drop=True)

    return df[["time", "open", "high", "low", "close", "volume", "log_return", *feature_cols]]


# The universe of instruments this project fetches/caches (see scripts/fetch_historical_data.py's
# DEFAULT_INSTRUMENTS) - "related instruments" for cross-pair features means "the other three of
# these", which is why RELATED_INSTRUMENT_COLUMNS below is a fixed list, not derived at runtime.
INSTRUMENT_UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def related_instruments(primary: str) -> list[str]:
    return [i for i in INSTRUMENT_UNIVERSE if i != primary]


# Every caller (live engine, backtest, RL training/validation) is expected to call
# compute_features with related_candles for related_instruments(instrument) and
# include_multi_timeframe=True (the default) - i.e. always produce this exact column set for
# that instrument. That consistency is what the "one feature pipeline, live and training"
# docstring promise actually depends on: a model trained on N columns fed M columns live is a
# silent, dangerous shape bug, not a graceful degradation.
#
# Now that multiple instruments are traded, this can no longer be one fixed global list - each
# instrument's related pairs (and therefore its xpair_* column names) differ. Every RL
# component (TradingEnv, RLPolicyStrategy, pretrain/train/tournament) takes feature_columns as
# an explicit constructor/call argument now rather than importing a constant, and
# app.persistence.models.ModelVersion.instrument records which instrument (and therefore which
# feature_columns_for(...) result) a saved model was trained against, so loading it later never
# has to guess.
_BASE_FEATURE_COLUMNS = [
    "momentum_return",
    "momentum_trend",
    "meanrev_zscore",
    "atr_pct",
    "volatility_regime",
    "rsi",
    "session_sin",
    "session_cos",
    "day_of_week",
    "carry_differential",
]


def feature_columns_for(instrument: str, *, include_multi_timeframe: bool = True) -> list[str]:
    """The exact column list compute_features(candles, related_candles=..., ...) produces for
    this instrument, in order - use this instead of hand-writing a column list anywhere a
    caller needs to know the shape of that instrument's feature/observation space."""
    multi_tf_cols = ["htf_trend_h4", "htf_trend_d"] if include_multi_timeframe else []
    cross_pair_cols = [f"xpair_{inst.lower()}_momentum" for inst in related_instruments(instrument)]
    return [*_BASE_FEATURE_COLUMNS, *multi_tf_cols, *cross_pair_cols]
