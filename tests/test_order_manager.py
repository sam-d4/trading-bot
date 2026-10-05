import datetime as dt

import pytest

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


def test_close_with_no_fill_leaves_trade_open_instead_of_recording_a_fake_close(db_session):
    """Regression: a close that never filled at OANDA was recorded locally as closed at entry price
    with zero P&L, while a 1.9M-unit position stayed open at OANDA."""
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    original_close = client.close_trade

    def close_without_fill(trade_id):
        result = original_close(trade_id)
        result.fill_price = None
        return result

    client.close_trade = close_without_fill
    om.flatten_all(db_session)

    assert len(repo.open_trades(db_session)) == 1


def test_reconcile_reopens_a_trade_recorded_closed_but_still_open_at_oanda(db_session):
    from app.execution.reconciliation import reconcile

    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    trade = repo.open_trades(db_session)[0]
    repo.record_trade_closed(
        db_session, oanda_trade_id=trade.oanda_trade_id, exit_price=1.10, closed_at=dt.datetime.now(dt.timezone.utc), realized_pnl=0.0
    )
    assert repo.open_trades(db_session) == []  # local says closed; the fake client still holds it open

    reconcile(db_session, client)

    assert [t.oanda_trade_id for t in repo.open_trades(db_session)] == [trade.oanda_trade_id]


def test_open_trade_is_sent_with_the_stop_distance_sizing_used(db_session):
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=-1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    assert client.placed_stops == [pytest.approx(0.0030)]  # 2 x ATR, always a positive distance


def test_signal_flip_does_not_open_new_trade_while_old_one_is_still_open_at_oanda(db_session):
    """If the close fails and the trade is genuinely still open, opening the opposite side would net
    against it - the new order must not be placed."""
    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    original_close = client.close_trade

    def close_without_fill(trade_id):
        result = original_close(trade_id)
        result.fill_price = None
        return result

    client.close_trade = close_without_fill
    result = om.execute_signal(
        db_session, instrument="EUR_USD", direction=-1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )

    assert result is None
    assert len(client.placed_orders) == 1  # only the original long


def test_signal_flip_after_oanda_already_stopped_the_trade_out_opens_the_new_trade(db_session):
    """The race server-side stops create: OANDA closed the trade, our row hasn't caught up, the
    model flips. TradeClose fails on the already-closed trade - that must reconcile the row and
    carry on, not leave the strategy stuck."""
    from app.broker.oanda_client import OandaClientError

    om, client, _ = _order_manager()
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    trade_id = next(iter(client._open_trades))
    client.remove_open_trade_remotely(trade_id, realized_pnl=-20.0, average_close_price=1.097)

    def close_fails(_trade_id):
        raise OandaClientError("trade already closed")

    client.close_trade = close_fails
    result = om.execute_signal(
        db_session, instrument="EUR_USD", direction=-1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )

    assert result is not None and result.status == "filled"
    assert len(client.placed_orders) == 2
    stopped = [t for t in repo.recent_trades(db_session) if t.oanda_trade_id == trade_id][0]
    assert stopped.status == TradeStatus.CLOSED and stopped.exit_price == 1.097
