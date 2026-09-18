import pytest

from app.risk.limits import RiskLimits
from app.risk.position_sizing import resolve_account_to_quote_rate, size_position

LIMITS = RiskLimits(
    max_daily_drawdown_pct=3.0,
    max_overall_drawdown_pct=10.0,
    risk_per_trade_pct=1.0,
    max_leverage=10.0,
    flatten_on_kill_switch=True,
)


def test_flat_direction_gives_zero_units():
    result = size_position(equity=10_000, price=1.10, atr=0.0015, direction=0, limits=LIMITS)
    assert result.units == 0.0


def test_long_position_sized_from_risk_pct_and_atr_stop():
    result = size_position(equity=10_000, price=1.10, atr=0.0015, direction=1, limits=LIMITS)
    # risk_amount = 10_000 * 1% = 100; stop_distance = 0.0015 * 2.0 = 0.003
    # raw_units = 100 / 0.003 = 33333.33...
    assert result.units > 0
    assert abs(result.units - 100 / 0.003) < 1e-6
    assert result.capped_by_leverage is False


def test_short_direction_gives_negative_units():
    result = size_position(equity=10_000, price=1.10, atr=0.0015, direction=-1, limits=LIMITS)
    assert result.units < 0


def test_leverage_cap_wins_when_volatility_is_very_low():
    # tiny ATR -> the risk-formula would demand a huge position; the leverage cap must clamp it.
    result = size_position(equity=10_000, price=1.10, atr=0.00001, direction=1, limits=LIMITS)
    max_units_from_leverage = (10_000 * LIMITS.max_leverage) / 1.10
    assert result.capped_by_leverage is True
    assert abs(result.units - max_units_from_leverage) < 1e-6


def test_zero_equity_gives_zero_units():
    result = size_position(equity=0, price=1.10, atr=0.0015, direction=1, limits=LIMITS)
    assert result.units == 0.0


def test_account_to_quote_rate_scales_units_proportionally():
    # GBP account (equity in GBP) trading EUR_USD (quote=USD) at a GBP/USD rate of 1.35:
    # the GBP risk budget is worth 1.35x more in USD, so it should buy 1.35x more EUR_USD units
    # for the same risk-in-quote-currency, everything else held equal.
    baseline = size_position(equity=10_000, price=1.10, atr=0.0015, direction=1, limits=LIMITS)
    converted = size_position(
        equity=10_000, price=1.10, atr=0.0015, direction=1, limits=LIMITS, account_to_quote_rate=1.35
    )
    assert converted.units == pytest.approx(baseline.units * 1.35)


def test_account_to_quote_rate_of_one_matches_no_conversion():
    baseline = size_position(equity=10_000, price=1.10, atr=0.0015, direction=1, limits=LIMITS)
    explicit = size_position(
        equity=10_000, price=1.10, atr=0.0015, direction=1, limits=LIMITS, account_to_quote_rate=1.0
    )
    assert explicit.units == baseline.units


def test_leverage_cap_also_scales_with_account_to_quote_rate():
    result = size_position(
        equity=10_000, price=1.10, atr=0.00001, direction=1, limits=LIMITS, account_to_quote_rate=1.35
    )
    max_units_from_leverage = (10_000 * 1.35 * LIMITS.max_leverage) / 1.10
    assert result.capped_by_leverage is True
    assert result.units == pytest.approx(max_units_from_leverage)


class TestResolveAccountToQuoteRate:
    def test_same_currency_needs_no_conversion(self):
        assert resolve_account_to_quote_rate("USD", "USD", {}) == 1.0

    def test_direct_pair_used_as_is(self):
        rate = resolve_account_to_quote_rate("GBP", "USD", {"GBP_USD": 1.35})
        assert rate == 1.35

    def test_inverse_pair_is_reciprocated(self):
        rate = resolve_account_to_quote_rate("USD", "GBP", {"GBP_USD": 1.25})
        assert rate == pytest.approx(1 / 1.25)

    def test_raises_when_no_rate_available(self):
        with pytest.raises(ValueError):
            resolve_account_to_quote_rate("CHF", "JPY", {"GBP_USD": 1.35})
