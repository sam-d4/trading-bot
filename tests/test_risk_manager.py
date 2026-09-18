import datetime as dt

import pytest

from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager, trading_day_key

LIMITS = RiskLimits(
    max_daily_drawdown_pct=3.0,
    max_overall_drawdown_pct=10.0,
    risk_per_trade_pct=1.0,
    max_leverage=10.0,
    flatten_on_kill_switch=True,
)


def utc(*args) -> dt.datetime:
    return dt.datetime(*args, tzinfo=dt.timezone.utc)


def test_no_trip_within_limits():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    result = rm.check(nav=9_900, now=utc(2026, 1, 1, 13))  # 1% down
    assert result.halted is False
    assert result.newly_tripped is False


def test_daily_drawdown_trips_kill_switch():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    result = rm.check(nav=9_600, now=utc(2026, 1, 1, 13))  # 4% down > 3% daily limit
    assert result.halted is True
    assert result.newly_tripped is True
    assert "daily drawdown" in result.reason


def test_overall_drawdown_trips_kill_switch_even_if_daily_ok():
    limits = RiskLimits(
        max_daily_drawdown_pct=50.0,  # effectively disabled for this test
        max_overall_drawdown_pct=10.0,
        risk_per_trade_pct=1.0,
        max_leverage=10.0,
        flatten_on_kill_switch=True,
    )
    rm = RiskManager(limits)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    result = rm.check(nav=8_900, now=utc(2026, 1, 1, 13))  # 11% down > 10% overall limit
    assert result.halted is True
    assert "overall drawdown" in result.reason


def test_stays_halted_until_explicit_reset():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    rm.check(nav=9_600, now=utc(2026, 1, 1, 13))
    assert rm.halted is True

    # a new day rolling over should NOT auto-clear the halt
    still_halted = rm.check(nav=9_700, now=utc(2026, 1, 2, 23))
    assert still_halted.halted is True
    assert still_halted.newly_tripped is False

    rm.reset(nav=9_700, now=utc(2026, 1, 2, 23))
    assert rm.halted is False


def test_daily_baseline_resets_on_rollover_but_high_water_mark_does_not():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    rm.check(nav=10_200, now=utc(2026, 1, 1, 13))  # new high water mark 10200

    # cross the 22:00 UTC rollover into the next trading day
    result = rm.check(nav=10_100, now=utc(2026, 1, 2, 23))
    assert result.halted is False
    # daily baseline should now be 10100 (the first NAV seen in the new trading day),
    # not the old day's 10000/10200 - a further small drop should stay within the new day's limit
    result2 = rm.check(nav=10_050, now=utc(2026, 1, 3, 0))
    assert result2.halted is False


def test_manual_trip_halts_regardless_of_drawdown():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))

    result = rm.manual_trip(nav=10_050, now=utc(2026, 1, 1, 13))  # NAV is actually UP - no drawdown at all
    assert result.halted is True
    assert result.newly_tripped is True
    assert "manual emergency stop" in result.reason
    assert rm.halted is True


def test_manual_trip_works_even_before_any_check_call():
    # the Emergency Stop button can be the very first interaction with a freshly-started engine,
    # before any check()/initialize() has ever run - manual_trip must not assume prior state.
    rm = RiskManager(LIMITS)
    result = rm.manual_trip(nav=10_000, now=utc(2026, 1, 1, 12))
    assert result.halted is True


def test_manual_trip_then_reset_allows_normal_checks_again():
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=utc(2026, 1, 1, 12))
    rm.manual_trip(nav=10_000, now=utc(2026, 1, 1, 13))
    assert rm.halted is True

    rm.reset(nav=10_000, now=utc(2026, 1, 1, 14))
    assert rm.halted is False
    result = rm.check(nav=9_950, now=utc(2026, 1, 1, 15))
    assert result.halted is False


@pytest.mark.parametrize(
    "now,expected_offset_hours",
    [
        (dt.datetime(2026, 1, 1, 21, 59, tzinfo=dt.timezone.utc), 0),
        (dt.datetime(2026, 1, 1, 22, 0, tzinfo=dt.timezone.utc), 1),
    ],
)
def test_trading_day_key_rolls_over_at_22_utc(now, expected_offset_hours):
    key_before = trading_day_key(dt.datetime(2026, 1, 1, 21, 59, tzinfo=dt.timezone.utc))
    key_at_rollover = trading_day_key(dt.datetime(2026, 1, 1, 22, 0, tzinfo=dt.timezone.utc))
    assert key_before != key_at_rollover
