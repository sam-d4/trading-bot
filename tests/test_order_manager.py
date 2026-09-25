import datetime as dt

from app.execution.order_manager import OrderManager
from app.persistence import repo
from app.persistence.models import TradeStatus
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager
from tests.fakes import FakeOandaClient

LIMITS = RiskLimits(
    max_daily_drawdown_pct=3.0,
    max_overall_drawdown_pct=10.0,
    risk_per_trade_pct=1.0,
    max_leverage=10.0,
    flatten_on_kill_switch=True,
)


def _order_manager(client=None, risk_manager=None) -> tuple[OrderManager, FakeOandaClient, RiskManager]:
    client = client or FakeOandaClient()
    rm = risk_manager or RiskManager(LIMITS)
    om = OrderManager(client, rm, LIMITS)
    return om, client, rm


def test_execute_signal_opens_a_trade(db_session):
    om, client, _ = _order_manager()
    result = om.execute_signal(
        db_session,
        instrument="EUR_USD",
        direction=1,
        price=1.10,
        atr=0.0015,
        equity=10_000,
        strategy_name="test",
    )
    assert result is not None
    assert result.status == "filled"
    open_trades = repo.open_trades(db_session)
    assert len(open_trades) == 1
    assert open_trades[0].instrument == "EUR_USD"


def test_flat_signal_with_no_position_does_nothing(db_session):
    om, client, _ = _order_manager()
    result = om.execute_signal(
        db_session, instrument="EUR_USD", direction=0, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert result is None
    assert client.placed_orders == []


def test_same_direction_signal_does_not_pyramid(db_session):
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.11, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert len(client.placed_orders) == 1
    assert len(repo.open_trades(db_session)) == 1


def test_reversed_signal_closes_then_reopens(db_session):
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=-1, price=1.09, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert len(client.closed_trade_ids) == 1
    assert len(client.placed_orders) == 2
    open_trades = repo.open_trades(db_session)
    assert len(open_trades) == 1
    assert open_trades[0].direction.value == "short"


def test_halted_risk_manager_refuses_new_orders(db_session):
    rm = RiskManager(LIMITS)
    rm.initialize(nav=10_000, now=dt.datetime.now(dt.timezone.utc))
    rm.check(nav=9_000, now=dt.datetime.now(dt.timezone.utc))  # trips overall drawdown
    assert rm.halted is True

    om, client, _ = _order_manager(risk_manager=rm)
    result = om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert result is None
    assert client.placed_orders == []


def test_flatten_all_closes_every_open_trade(db_session):
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert len(repo.open_trades(db_session)) == 1

    om.flatten_all(db_session)
    assert len(repo.open_trades(db_session)) == 0
    closed = repo.recent_trades(db_session)
    assert closed[0].status == TradeStatus.CLOSED


def test_closed_trade_records_oandas_account_currency_pl_not_quote_currency_pnl(db_session):
    """Regression: (exit - entry) * units is in the pair's QUOTE currency (JPY for USD_JPY), not the
    account's - a small JPY loss was recorded as a large account-currency loss."""
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    original_close = client.close_trade

    def close_with_account_pl(trade_id):
        result = original_close(trade_id)
        result.raw = {"orderFillTransaction": {"pl": "-3.25"}}
        return result

    client.close_trade = close_with_account_pl
    om.flatten_all(db_session)

    assert repo.recent_trades(db_session)[0].realized_pnl == -3.25
