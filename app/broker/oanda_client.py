"""Thin wrapper around oandapyV20 for the account/order operations the bot needs.

Kept deliberately narrow (account summary, market orders, open trades, close trade) rather than
exposing the whole OANDA v20 surface - callers should not need to know oandapyV20's request/response
shapes, only this module's typed methods.
"""

import structlog
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountInstruments, AccountSummary as AccountSummaryEndpoint
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import OpenTrades, TradeClose, TradeDetails
from oandapyV20.exceptions import V20Error

from app.broker.models import AccountSummary, OpenTrade, OrderResult
from app.config.settings import Settings

log = structlog.get_logger(__name__)


class OandaClientError(RuntimeError):
    pass


class OandaClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._account_id = settings.oanda_account_id
        environment = "practice" if settings.oanda_environment == "practice" else "live"
        self._api = API(access_token=settings.oanda_api_token, environment=environment, request_params={"timeout": 15})

    def get_account_summary(self) -> AccountSummary:
        req = AccountSummaryEndpoint(accountID=self._account_id)
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"failed to fetch account summary: {exc}") from exc

        acct = resp["account"]
        return AccountSummary(
            account_id=acct["id"],
            currency=acct["currency"],
            balance=float(acct["balance"]),
            nav=float(acct["NAV"]),
            unrealized_pl=float(acct["unrealizedPL"]),
            margin_used=float(acct["marginUsed"]),
            margin_available=float(acct["marginAvailable"]),
            open_trade_count=int(acct["openTradeCount"]),
            open_position_count=int(acct["openPositionCount"]),
        )

    def get_financing_rates(self, instruments: list[str]) -> dict[str, float]:
        """Real, live carry rates from OANDA's own books - annualized long-vs-short financing
        spread per instrument (longRate - shortRate), used as app/data/features.py's
        carry_differential. This is the CURRENT rate only; OANDA doesn't expose a financing-rate
        history, so a backtest over past years necessarily applies today's rate as a constant
        across the whole window rather than the true (materially different, e.g. pre- vs
        post-2022-hiking-cycle) historical differential - a real approximation, not a bug, but
        one callers using this for historical backtesting should be aware of."""
        req = AccountInstruments(accountID=self._account_id, params={"instruments": ",".join(instruments)})
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"failed to fetch financing rates: {exc}") from exc

        rates = {}
        for inst in resp.get("instruments", []):
            financing = inst.get("financing", {})
            if "longRate" in financing and "shortRate" in financing:
                rates[inst["name"]] = float(financing["longRate"]) - float(financing["shortRate"])
        return rates

    def get_open_trades(self) -> list[OpenTrade]:
        req = OpenTrades(accountID=self._account_id)
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"failed to fetch open trades: {exc}") from exc

        return [
            OpenTrade(
                trade_id=t["id"],
                instrument=t["instrument"],
                units=float(t["currentUnits"]),
                price=float(t["price"]),
                unrealized_pl=float(t["unrealizedPL"]),
                open_time=t["openTime"],
            )
            for t in resp.get("trades", [])
        ]

    def place_market_order(self, instrument: str, units: float) -> OrderResult:
        """units: positive to go/add long, negative to go/add short. Sizing is decided upstream
        by the risk manager - this method just executes what it's told."""
        order_payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        req = OrderCreate(accountID=self._account_id, data=order_payload)
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"order create failed: {exc}") from exc

        if "orderCancelTransaction" in resp:
            reason = resp["orderCancelTransaction"].get("reason", "unknown")
            log.warning("order_cancelled", instrument=instrument, units=units, reason=reason)
            return OrderResult(order_id=None, trade_id=None, fill_price=None, status="cancelled", raw=resp)

        fill = resp.get("orderFillTransaction")
        if fill is None:
            return OrderResult(
                order_id=resp.get("orderCreateTransaction", {}).get("id"),
                trade_id=None,
                fill_price=None,
                status="pending",
                raw=resp,
            )

        return OrderResult(
            order_id=fill.get("orderID"),
            trade_id=fill.get("tradeOpened", {}).get("tradeID"),
            fill_price=float(fill["price"]) if "price" in fill else None,
            status="filled",
            raw=resp,
        )

    def get_trade(self, trade_id: str) -> dict:
        """Raw trade detail dict (not wrapped in a pydantic model) - used by
        execution/reconciliation.py to look up the final state of a trade that's disappeared
        from the open-trades list. Field availability for a CLOSED trade isn't fully verified
        against a live account yet - callers should read defensively (.get with fallbacks)."""
        req = TradeDetails(accountID=self._account_id, tradeID=trade_id)
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"failed to fetch trade {trade_id}: {exc}") from exc
        return resp["trade"]

    def close_trade(self, trade_id: str) -> OrderResult:
        req = TradeClose(accountID=self._account_id, tradeID=trade_id)
        try:
            resp = self._api.request(req)
        except V20Error as exc:
            raise OandaClientError(f"trade close failed: {exc}") from exc

        fill = resp.get("orderFillTransaction", {})
        return OrderResult(
            order_id=fill.get("orderID"),
            trade_id=trade_id,
            fill_price=float(fill["price"]) if "price" in fill else None,
            status="closed",
            raw=resp,
        )
