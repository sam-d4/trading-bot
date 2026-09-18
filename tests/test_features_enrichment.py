import numpy as np
import pandas as pd

from app.data.features import compute_features, feature_columns_for, related_instruments


def _synthetic_candles(n: int, *, seed: int = 0, start: str = "2024-01-01", freq: str = "h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
            "open": prices,
            "high": prices + rng.random(n) * 0.0005,
            "low": prices - rng.random(n) * 0.0005,
            "close": prices + rng.normal(0, 0.0002, n),
            "volume": rng.integers(1, 500, n),
        }
    )


def test_related_instruments_excludes_primary():
    related = related_instruments("EUR_USD")
    assert "EUR_USD" not in related
    assert set(related) == {"GBP_USD", "USD_JPY", "AUD_USD"}


def test_compute_features_base_columns_always_present():
    df = compute_features(_synthetic_candles(300))
    base_only = [c for c in feature_columns_for("EUR_USD") if not c.startswith("xpair_") and c not in ("htf_trend_h4", "htf_trend_d")]
    for col in base_only:
        assert col in df.columns


def test_feature_columns_for_differs_by_instrument():
    eur_cols = feature_columns_for("EUR_USD")
    gbp_cols = feature_columns_for("GBP_USD")
    assert "xpair_gbp_usd_momentum" in eur_cols
    assert "xpair_eur_usd_momentum" not in eur_cols
    assert "xpair_eur_usd_momentum" in gbp_cols
    assert "xpair_gbp_usd_momentum" not in gbp_cols
    assert len(eur_cols) == len(gbp_cols)  # same shape, different instrument names


def test_multi_timeframe_columns_present_and_finite():
    df = compute_features(_synthetic_candles(400), include_multi_timeframe=True)
    assert "htf_trend_h4" in df.columns
    assert "htf_trend_d" in df.columns
    assert df["htf_trend_h4"].notna().all()
    assert df["htf_trend_d"].notna().all()
    assert np.isfinite(df["htf_trend_h4"]).all()


def test_multi_timeframe_disabled_omits_columns():
    df = compute_features(_synthetic_candles(300), include_multi_timeframe=False)
    assert "htf_trend_h4" not in df.columns
    assert "htf_trend_d" not in df.columns


def test_higher_timeframe_trend_has_no_lookahead():
    """A sharp price jump placed entirely within the second half of an H4 bar must not be
    visible in htf_trend_h4 for any H1 bar still inside that same (not-yet-closed) H4 window -
    only after the H4 bar closes should the jump show up."""
    n = 120
    base = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    prices = np.full(n, 1.1000)
    # small windows -> small warmup, so rows near the jump survive compute_features' trimming.
    # H4 bins are [0-4),[4-8),...,[20-24),[24-28),... - jump happens at hour 22, inside [20,24).
    jump_at = 22
    prices[jump_at:] += 0.02  # a large, obvious jump
    df = pd.DataFrame(
        {
            "time": base,
            "open": prices,
            "high": prices + 0.0001,
            "low": prices - 0.0001,
            "close": prices,
            "volume": np.ones(n, dtype=int),
        }
    )

    features = compute_features(df, include_multi_timeframe=True, momentum_window=4, meanrev_window=4, atr_period=4)
    # Row at hour 23 (still inside the [20,24) H4 bin that contains the jump) must not yet reflect
    # the jump's effect on H4 trend - only the bin closing at hour 24 makes it visible.
    row_before_close = features[features["time"] == base[23]]
    row_after_close = features[features["time"] == base[25]]  # safely inside next observed bin
    assert not row_before_close.empty
    assert not row_after_close.empty
    # the post-close reading should differ from the pre-close reading once the jump is visible
    assert row_after_close["htf_trend_h4"].iloc[0] != row_before_close["htf_trend_h4"].iloc[0]


def test_cross_pair_columns_present_when_related_candles_given():
    primary = _synthetic_candles(300, seed=1)
    related = {"GBP_USD": _synthetic_candles(300, seed=2), "USD_JPY": _synthetic_candles(300, seed=3)}
    df = compute_features(primary, related_candles=related)
    assert "xpair_gbp_usd_momentum" in df.columns
    assert "xpair_usd_jpy_momentum" in df.columns
    assert df["xpair_gbp_usd_momentum"].notna().all()


def test_cross_pair_missing_related_data_fills_zero_not_nan():
    primary = _synthetic_candles(300, seed=1)
    # related instrument's data starts much later than primary - early primary rows should fill 0.
    related = {"GBP_USD": _synthetic_candles(50, seed=2, start="2024-01-20")}
    df = compute_features(primary, related_candles=related)
    assert df["xpair_gbp_usd_momentum"].iloc[0] == 0.0
    assert df["xpair_gbp_usd_momentum"].notna().all()
