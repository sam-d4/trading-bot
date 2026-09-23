import datetime as dt

import pandas as pd
import pytest

import app.core.engine as engine_module
from app.broker.models import Tick
from app.config.settings import Settings
from app.core.engine import BAR_HISTORY_LEN, TradingEngine
from app.execution.order_manager import OrderManager
from app.notifications.alerts import AlertSender
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager
from app.strategy.baselines import TrendFollowingStrategy
from tests.fakes import FakeOandaClient


def _engine(bar_seconds: int = 60) -> TradingEngine:
    settings = Settings(oanda_api_token="x", oanda_account_id="y", live_bar_seconds=bar_seconds)
    client = FakeOandaClient()
    limits = RiskLimits(
        max_daily_drawdown_pct=3.0,
        max_overall_drawdown_pct=10.0,
        risk_per_trade_pct=1.0,
        max_leverage=10.0,
        flatten_on_kill_switch=True,
    )
    risk_manager = RiskManager(limits)
    order_manager = OrderManager(client, risk_manager, limits)
    alerts = AlertSender(settings)
    return TradingEngine(settings, client, {"EUR_USD": TrendFollowingStrategy()}, risk_manager, order_manager, alerts)


def _tick(iso_time: str, mid: float) -> Tick:
    return Tick(instrument="EUR_USD", time=iso_time, bid=mid - 0.0001, ask=mid + 0.0001)


def test_ticks_within_same_bucket_update_one_bar():
    engine = _engine(bar_seconds=60)
    engine._on_tick(_tick("2026-01-01T00:00:05Z", 1.1000))
    engine._on_tick(_tick("2026-01-01T00:00:20Z", 1.1010))
    engine._on_tick(_tick("2026-01-01T00:00:40Z", 1.0990))

    current = engine._current_bar["EUR_USD"]
    assert current["open"] == 1.1000
    assert current["high"] == 1.1010
    assert current["low"] == 1.0990
    assert current["close"] == 1.0990
    assert current["volume"] == 3
    assert len(engine._bars["EUR_USD"]) == 0  # bar not finalized yet


def test_tick_in_next_bucket_finalizes_previous_bar():
    engine = _engine(bar_seconds=60)
    engine._on_tick(_tick("2026-01-01T00:00:05Z", 1.1000))
    engine._on_tick(_tick("2026-01-01T00:00:20Z", 1.1010))
    engine._on_tick(_tick("2026-01-01T00:01:05Z", 1.1020))  # crosses into the next 60s bucket

    assert len(engine._bars["EUR_USD"]) == 1
    finalized = engine._bars["EUR_USD"][0]
    assert finalized["open"] == 1.1000
    assert finalized["high"] == 1.1010
    assert finalized["close"] == 1.1010
    # the new bucket's bar should have started fresh with the crossing tick
    current = engine._current_bar["EUR_USD"]
    assert current["open"] == 1.1020
    assert current["volume"] == 1


def test_many_bucket_crossings_produce_one_finalized_bar_each():
    engine = _engine(bar_seconds=60)
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(5):
        t = (base + dt.timedelta(seconds=i * 61)).strftime("%Y-%m-%dT%H:%M:%SZ")
        engine._on_tick(_tick(t, 1.1000 + i * 0.0001))

    # 5 ticks each in a distinct bucket -> 4 finalized bars, 1 still open
    assert len(engine._bars["EUR_USD"]) == 4


def test_related_instrument_ticks_are_tracked_but_never_traded():
    engine = _engine(bar_seconds=60)
    gbp_tick = Tick(instrument="GBP_USD", time="2026-01-01T00:00:05Z", bid=1.2999, ask=1.3001)
    engine._on_tick(gbp_tick)

    assert engine._current_bar["GBP_USD"]["open"] == 1.3000
    assert engine._current_bar["EUR_USD"] is None  # untouched - no EUR_USD tick sent yet


def test_resolve_account_to_quote_rate_uses_tracked_gbp_usd_bar():
    engine = _engine(bar_seconds=60)
    # GBP_USD bar finalizes once a second tick lands in a later bucket
    engine._on_tick(Tick(instrument="GBP_USD", time="2026-01-01T00:00:05Z", bid=1.3499, ask=1.3501))
    engine._on_tick(Tick(instrument="GBP_USD", time="2026-01-01T00:01:05Z", bid=1.3499, ask=1.3501))

    rate = engine._resolve_account_to_quote_rate("EUR_USD", "GBP")  # EUR_USD's quote currency is USD
    assert rate == pytest.approx(1.35, abs=1e-6)


def test_resolve_account_to_quote_rate_needs_no_conversion_when_currencies_match():
    engine = _engine(bar_seconds=60)
    rate = engine._resolve_account_to_quote_rate("EUR_USD", "USD")  # EUR_USD's quote currency is already USD
    assert rate == 1.0


def test_resolve_account_to_quote_rate_falls_back_to_one_when_unavailable():
    engine = _engine(bar_seconds=60)
    # no bars tracked for any currency pair yet - GBP has no route to USD in rate_lookup
    rate = engine._resolve_account_to_quote_rate("EUR_USD", "GBP")
    assert rate == 1.0


async def test_seed_bar_history_fills_bars_past_the_finalize_warmup_gate(monkeypatch):
    """Regression test for a real bug: self._bars starts empty every process restart, and
    _finalize_bar refuses to call strategy.decide() until BAR_HISTORY_LEN // 2 bars have
    accumulated - purely from live ticks, that took over 8 days, and this project's live engine
    had never once run that long uninterrupted. _seed_bar_history must clear that gate
    immediately from OANDA's own recent-candle history, without needing any live ticks first."""
    engine = _engine(bar_seconds=60)

    def fake_fetch_candles(settings, instrument, granularity, start, end):
        rows = [
            {"time": start + dt.timedelta(hours=i), "open": 1.1, "high": 1.1005, "low": 1.0995, "close": 1.1, "volume": 10}
            for i in range(BAR_HISTORY_LEN + 50)
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr(engine_module, "fetch_candles", fake_fetch_candles)

    await engine._seed_bar_history()

    assert len(engine._bars["EUR_USD"]) == BAR_HISTORY_LEN  # capped by the deque's maxlen
    assert len(engine._bars["EUR_USD"]) >= BAR_HISTORY_LEN // 2  # clears _finalize_bar's warmup gate
