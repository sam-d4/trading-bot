"""Phase 1 verification script.

Proves the OANDA plumbing works end to end against the configured account (practice by default):
  1. fetch and print the account summary
  2. place a tiny market order (1 unit of the primary instrument)
  3. immediately close it

Market orders fill-or-kill immediately, so there's nothing to "cancel" pre-fill - the equivalent
round-trip proof is open-then-close, which is what this does. Refuses to touch a live account
unless you pass --i-understand-this-is-live, as a deliberate speed bump against running it by
habit once OANDA_ENVIRONMENT is ever switched to "live".
"""

import argparse
import sys

from app.broker.oanda_client import OandaClient, OandaClientError
from app.config.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-understand-this-is-live",
        action="store_true",
        help="required to proceed if OANDA_ENVIRONMENT=live",
    )
    parser.add_argument(
        "--skip-order-test",
        action="store_true",
        help="only fetch account summary, don't place/close a test order",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not settings.oanda_api_token or not settings.oanda_account_id:
        print("OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set. Copy .env.example to .env and fill them in.")
        return 1

    if settings.is_live and not args.i_understand_this_is_live:
        print(
            "OANDA_ENVIRONMENT=live and --i-understand-this-is-live was not passed. "
            "Refusing to place a test order against a live account. Aborting."
        )
        return 1

    client = OandaClient(settings)

    print(f"Environment: {settings.oanda_environment} ({settings.oanda_rest_host})")
    try:
        summary = client.get_account_summary()
    except OandaClientError as exc:
        print(f"FAILED to fetch account summary: {exc}")
        return 1

    print("Account summary:")
    print(f"  account_id            {summary.account_id}")
    print(f"  currency              {summary.currency}")
    print(f"  balance               {summary.balance}")
    print(f"  NAV                   {summary.nav}")
    print(f"  unrealized P/L        {summary.unrealized_pl}")
    print(f"  margin used/available {summary.margin_used} / {summary.margin_available}")
    print(f"  open trades/positions {summary.open_trade_count} / {summary.open_position_count}")

    if args.skip_order_test:
        return 0

    instrument = settings.primary_instrument
    print(f"\nPlacing a 1-unit test market order on {instrument}...")
    try:
        order = client.place_market_order(instrument, units=1)
    except OandaClientError as exc:
        print(f"FAILED to place test order: {exc}")
        return 1

    if order.status != "filled" or not order.trade_id:
        print(f"Order did not fill as expected: status={order.status} raw={order.raw}")
        return 1

    print(f"  filled: trade_id={order.trade_id} price={order.fill_price}")

    print(f"Closing trade {order.trade_id}...")
    try:
        close = client.close_trade(order.trade_id)
    except OandaClientError as exc:
        print(f"FAILED to close test trade: {exc} -- you may need to close it manually in OANDA.")
        return 1

    print(f"  closed: price={close.fill_price}")
    print("\nRound trip OK: account summary + order place + close all succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
