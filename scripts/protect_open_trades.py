"""Attach a server-side stop-loss to every open OANDA trade that doesn't have one.

New trades get their stop at order time (OandaClient.place_market_order's stop_distance); this
covers trades opened before that existed, or adopted by reconciliation without one.

Stop price = 2 x ATR (the same distance position sizing assumes) from whichever is TIGHTER of the
trade's entry price and the current price: for a trade in profit that locks most of it in instead
of letting it round-trip, and for a trade at a loss it is never worse than the original stop.

Usage:
    python scripts/protect_open_trades.py --dry-run
    python scripts/protect_open_trades.py
"""

import argparse
import datetime as dt
import sys

from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingInfo

from app.broker.oanda_client import OandaClient, OandaClientError, format_price
from app.config.settings import get_settings
from app.data.candles import fetch_candles
from app.data.features import compute_features
from app.risk.position_sizing import ATR_STOP_MULTIPLIER


def current_mid(settings, instrument: str) -> float:
    api = API(access_token=settings.oanda_api_token, environment="practice" if not settings.is_live else "live")
    resp = api.request(PricingInfo(accountID=settings.oanda_account_id, params={"instruments": instrument}))
    price = resp["prices"][0]
    return (float(price["bids"][0]["price"]) + float(price["asks"][0]["price"])) / 2


def atr_for(settings, instrument: str) -> float:
    now = dt.datetime.now(dt.timezone.utc)
    candles = fetch_candles(settings, instrument, "H1", now - dt.timedelta(days=30), now)
    row = compute_features(candles).iloc[-1]
    return float(row["atr_pct"] * row["close"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print what would be set, change nothing")
    parser.add_argument("--i-understand-this-is-live", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if settings.is_live and not args.i_understand_this_is_live:
        print("OANDA_ENVIRONMENT=live and --i-understand-this-is-live was not passed. Aborting.")
        return 1

    client = OandaClient(settings)
    open_trades = client.get_open_trades()
    if not open_trades:
        print("No open trades.")
        return 0

    for trade in open_trades:
        detail = client.get_trade(trade.trade_id)
        existing = detail.get("stopLossOrder")
        if existing:
            print(f"trade {trade.trade_id} {trade.instrument}: already has a stop at {existing.get('price')} - leaving it")
            continue

        distance = ATR_STOP_MULTIPLIER * atr_for(settings, trade.instrument)
        mid = current_mid(settings, trade.instrument)
        if trade.units > 0:  # long: stop below
            stop = max(trade.price - distance, mid - distance)
        else:  # short: stop above
            stop = min(trade.price + distance, mid + distance)

        print(
            f"trade {trade.trade_id} {trade.instrument} units={trade.units:.0f} entry={trade.price} "
            f"now={mid:.5f} atr_distance={distance:.5f} -> stop {format_price(trade.instrument, stop)}"
        )
        if args.dry_run:
            continue
        try:
            client.set_trade_stop_loss(trade.trade_id, trade.instrument, stop)
        except OandaClientError as exc:
            print(f"  FAILED: {exc}")
            return 1
        confirmed = client.get_trade(trade.trade_id).get("stopLossOrder")
        print(f"  stop attached: {confirmed.get('price') if confirmed else 'NOT FOUND ON READ-BACK'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
