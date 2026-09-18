from app.execution.order_manager import OrderManager
from app.execution.reconciliation import reconcile
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


def test_reconcile_closes_locally_open_trade_that_oanda_closed_remotely(db_session):
    client = FakeOandaClient()
    om = OrderManager(client, RiskManager(LIMITS), LIMITS)
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )
    trade_id = next(iter(client._open_trades))

    client.remove_open_trade_remotely(trade_id, realized_pnl=42.0)

    reconcile(db_session, client)

    assert len(repo.open_trades(db_session)) == 0
    closed = repo.recent_trades(db_session)[0]
    assert closed.status == TradeStatus.CLOSED
    assert closed.realized_pnl == 42.0


def test_reconcile_leaves_trade_alone_if_oanda_still_reports_it_open(db_session):
    client = FakeOandaClient()
    om = OrderManager(client, RiskManager(LIMITS), LIMITS)
    om.execute_signal(
        db_session, instrument="EUR_USD", direction=1, price=1.10, atr=0.0015, equity=10_000, strategy_name="test"
    )

    reconcile(db_session, client)

    assert len(repo.open_trades(db_session)) == 1


def test_reconcile_adopts_untracked_remote_open_trade(db_session):
    client = FakeOandaClient()
    client.add_untracked_open_trade("999", "EUR_USD", 1000, 1.12)

    assert len(repo.open_trades(db_session)) == 0

    reconcile(db_session, client)

    open_trades = repo.open_trades(db_session)
    assert len(open_trades) == 1
    assert open_trades[0].oanda_trade_id == "999"
    assert open_trades[0].strategy_name == "untracked_adopted"


def test_reconcile_handles_fetch_failure_gracefully(db_session):
    class BrokenClient(FakeOandaClient):
        def get_open_trades(self):
            from app.broker.oanda_client import OandaClientError

            raise OandaClientError("network down")

    client = BrokenClient()
    reconcile(db_session, client)  # should not raise
    assert len(repo.open_trades(db_session)) == 0
