"""Fetch and cache multi-year historical candles for one or more instruments.

This is the raw material for backtesting the baseline strategies and, later, for pretraining
the RL policy (see the plan's "cold-start knowledge" section - deep, multi-pair history matters
more here than for a strategy that's only ever fine-tuned on recent data).

Usage:
    python scripts/fetch_historical_data.py --years 5
    python scripts/fetch_historical_data.py --instruments EUR_USD GBP_USD USD_JPY --years 10
"""

import argparse
import datetime as dt
import sys

from app.config.settings import get_settings
from app.data.candles import CandleFetchError, fetch_candles
from app.data.store import save_candles

DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", nargs="+", default=DEFAULT_INSTRUMENTS)
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.oanda_api_token or not settings.oanda_account_id:
        print("OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set. Copy .env.example to .env and fill them in.")
        return 1

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=365 * args.years)

    for instrument in args.instruments:
        print(f"Fetching {instrument} {args.granularity} from {start.date()} to {end.date()}...")
        try:
            df = fetch_candles(settings, instrument, args.granularity, start, end)
        except CandleFetchError as exc:
            print(f"  FAILED: {exc}")
            continue

        if df.empty:
            print("  no candles returned")
            continue

        path = save_candles(df, instrument, args.granularity)
        print(f"  saved {len(df)} candles -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
