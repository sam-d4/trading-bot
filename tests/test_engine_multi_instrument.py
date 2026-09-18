"""Verifies the engine actually trades several instruments concurrently, independently - the
core new capability behind trading more than just the primary pair. Each traded instrument must
get its own strategy decisions and its own orders, without one instrument's bar completing
triggering a decision (or consuming) another's."""

import datetime as dt

import pytest

from app.broker.models import Tick
from app.config.settings import Settings
from app.core.engine import TradingEngine
from app.execution.order_manager import OrderManager
from app.notifications.alerts import AlertSender
from app.persistence.db import init_db
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager
from app.strategy.base import Signal, Strategy
from tests.fakes import FakeOandaClient

# The engine's _act_on_signal uses app.persistence.db's module-level SessionLocal directly (not
# an injected session, unlike most other code here) - its tables are otherwise never created in
# a test run, since every other test uses conftest.py's isolated db_session fixture engine
# instead. These are the first tests to exercise that real path end-to-end.
init_db()


@pytest.fixture(autouse=True)
def _clean_module_level_trades_table():
    """The module-level DB (SQLite :memory:, SingletonThreadPool) persists for the whole pytest
    process, not per-test - without this, an open trade left by one test looks like an existing
    position to the next test's OrderManager, silently skipping its order as "already
    positioned" rather than opening a fresh one."""
    from app.persistence.db import SessionLocal
    from app.persistence.models import Trade

    with SessionLocal() as session:
        session.query(Trade).delete()
        session.commit()
    yield

LIMITS = RiskLimits(
    max_daily_drawdown_pct=3.0,
    max_overall_drawdown_pct=10.0,
    risk_per_trade_pct=1.0,
    max_leverage=10.0,
    flatten_on_kill_switch=True,
)


class RecordingStrategy(Strategy):
    """Always signals LONG and records every features_row it was asked to decide on, so the
    test can confirm each traded instrument's strategy is invoked independently."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list = []

    def decide(self, features_row) -> Signal:
        self.calls.append(features_row["close"])
        return Signal.LONG


def _multi_engine(traded: dict[str, Strategy]) -> tuple[TradingEngine, FakeOandaClient]:
    settings = Settings(oanda_api_token="x", oanda_account_id="y", live_bar_seconds=60)
    client = FakeOandaClient()
    risk_manager = RiskManager(LIMITS)
    order_manager = OrderManager(client, risk_manager, LIMITS)
    alerts = AlertSender(settings)
    engine = TradingEngine(settings, client, traded, risk_manager, order_manager, alerts)
    return engine, client


def _feed_enough_bars_to_trigger_decisions(engine: TradingEngine, instrument: str, base_price: float) -> None:
    """_finalize_bar only starts making decisions once BAR_HISTORY_LEN // 2 bars exist - feed
    that many one-bar-per-tick, cheaply, without needing real market data."""
    from app.core.engine import BAR_HISTORY_LEN

    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    needed_bars = BAR_HISTORY_LEN // 2 + 1
    for i in range(needed_bars):
        t = (base + dt.timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        price = base_price + i * 0.00001
        engine._on_tick(Tick(instrument=instrument, time=t, bid=price - 0.0001, ask=price + 0.0001))


def test_two_traded_instruments_each_get_independent_strategy_decisions():
    eur_strategy = RecordingStrategy("eur_test")
    gbp_strategy = RecordingStrategy("gbp_test")
    engine, client = _multi_engine({"EUR_USD": eur_strategy, "GBP_USD": gbp_strategy})

    _feed_enough_bars_to_trigger_decisions(engine, "EUR_USD", 1.1000)

    assert len(eur_strategy.calls) > 0
    assert len(gbp_strategy.calls) == 0  # no GBP_USD ticks sent yet - its strategy must stay untouched

    _feed_enough_bars_to_trigger_decisions(engine, "GBP_USD", 1.3000)
    assert len(gbp_strategy.calls) > 0


def test_two_traded_instruments_place_orders_on_their_own_instrument():
    eur_strategy = RecordingStrategy("eur_test")
    gbp_strategy = RecordingStrategy("gbp_test")
    engine, client = _multi_engine({"EUR_USD": eur_strategy, "GBP_USD": gbp_strategy})

    _feed_enough_bars_to_trigger_decisions(engine, "EUR_USD", 1.1000)
    _feed_enough_bars_to_trigger_decisions(engine, "GBP_USD", 1.3000)

    traded_instruments = {inst for inst, _ in client.placed_orders}
    assert "EUR_USD" in traded_instruments
    assert "GBP_USD" in traded_instruments


def test_related_context_instrument_never_gets_traded_even_with_two_traded_pairs():
    engine, client = _multi_engine({"EUR_USD": RecordingStrategy("eur"), "GBP_USD": RecordingStrategy("gbp")})

    _feed_enough_bars_to_trigger_decisions(engine, "USD_JPY", 150.00)  # context-only, not traded

    assert client.placed_orders == []
    assert "USD_JPY" in engine._bars  # still tracked, for cross-pair context
