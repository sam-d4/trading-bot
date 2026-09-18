"""In-memory fakes for the OANDA client, used to unit-test execution/risk logic without any
network access or real credentials."""

from app.broker.models import AccountSummary, OpenTrade, OrderResult


class FakeOandaClient:
    def __init__(self, *, nav: float = 10_000.0, balance: float = 10_000.0):
        self.nav = nav
        self.balance = balance
        self.unrealized_pl = 0.0
        self._open_trades: dict[str, OpenTrade] = {}
        self._next_trade_id = 1
        self.placed_orders: list[tuple[str, float]] = []
        self.closed_trade_ids: list[str] = []
        self.trade_details: dict[str, dict] = {}

    def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            account_id="fake-account",
            currency="USD",
            balance=self.balance,
            nav=self.nav,
            unrealized_pl=self.unrealized_pl,
            margin_used=0.0,
            margin_available=self.nav,
            open_trade_count=len(self._open_trades),
            open_position_count=len(self._open_trades),
        )

    def get_open_trades(self) -> list[OpenTrade]:
        return list(self._open_trades.values())

    def place_market_order(self, instrument: str, units: float) -> OrderResult:
        self.placed_orders.append((instrument, units))
        trade_id = str(self._next_trade_id)
        self._next_trade_id += 1
        fill_price = 1.1000
        self._open_trades[trade_id] = OpenTrade(
            trade_id=trade_id,
            instrument=instrument,
            units=units,
            price=fill_price,
            unrealized_pl=0.0,
            open_time="2026-01-01T00:00:00Z",
        )
        return OrderResult(order_id=trade_id, trade_id=trade_id, fill_price=fill_price, status="filled", raw={})

    def close_trade(self, trade_id: str) -> OrderResult:
        self.closed_trade_ids.append(trade_id)
        trade = self._open_trades.pop(trade_id, None)
        fill_price = trade.price + 0.0010 if trade else 1.1010
        return OrderResult(order_id=trade_id, trade_id=trade_id, fill_price=fill_price, status="closed", raw={})

    def get_trade(self, trade_id: str) -> dict:
        return self.trade_details.get(trade_id, {"id": trade_id, "state": "CLOSED", "realizedPL": "0.0"})

    def get_financing_rates(self, instruments: list[str]) -> dict[str, float]:
        return dict.fromkeys(instruments, 0.0)

    # --- test helpers, not part of the real OandaClient interface -----------------------------

    def remove_open_trade_remotely(self, trade_id: str, *, realized_pnl: float = 0.0) -> None:
        """Simulates OANDA closing a trade on its own (stop-out, margin closeout) without going
        through place_market_order/close_trade - used to test reconciliation."""
        self._open_trades.pop(trade_id, None)
        self.trade_details[trade_id] = {"id": trade_id, "state": "CLOSED", "realizedPL": str(realized_pnl)}

    def add_untracked_open_trade(self, trade_id: str, instrument: str, units: float, price: float) -> None:
        """Simulates a trade OANDA has open that this process never recorded locally (e.g. after
        a crash mid-order) - used to test reconciliation's adoption path."""
        self._open_trades[trade_id] = OpenTrade(
            trade_id=trade_id, instrument=instrument, units=units, price=price,
            unrealized_pl=0.0, open_time="2026-01-01T00:00:00Z",
        )
