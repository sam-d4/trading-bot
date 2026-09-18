"""Backtest the baseline strategy library against cached historical data and print a comparison
table against buy-and-hold. Run scripts/fetch_historical_data.py first (for the primary
instrument AND its related pairs, if you want cross-pair features - the default fetch already
covers all four).

Usage:
    python scripts/run_backtest.py --instrument EUR_USD --granularity H1
"""

import argparse
import sys

from app.backtest.engine import buy_and_hold_result, run_backtest
from app.config.settings import get_settings
from app.data.enrichment import load_feature_context
from app.data.features import compute_features
from app.data.store import load_candles
from app.strategy.baselines import default_baseline_strategies

METRIC_ORDER = ["total_return_pct", "sharpe", "max_drawdown_pct", "win_rate", "profit_factor", "trade_count"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    args = parser.parse_args()

    try:
        candles = load_candles(args.instrument, args.granularity)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    settings = get_settings()
    related, carry_differential = load_feature_context(settings, args.instrument, args.granularity)
    print(f"Cross-pair data: {list(related.keys()) or 'none cached'}")
    print(f"Live carry differential for {args.instrument}: {carry_differential:.4f} (current OANDA rate, held constant across the whole backtest)")

    features = compute_features(candles, related_candles=related, carry_differential=carry_differential)
    print(f"{len(features)} feature rows for {args.instrument} {args.granularity}\n")

    rows = []
    bh = buy_and_hold_result(features, granularity=args.granularity)
    rows.append(("buy_and_hold", bh.metrics))

    for strategy in default_baseline_strategies():
        result = run_backtest(
            features, strategy, granularity=args.granularity, transaction_cost_bps=args.transaction_cost_bps
        )
        rows.append((strategy.name, result.metrics))

    header = f"{'strategy':<28}" + "".join(f"{m:>16}" for m in METRIC_ORDER)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        print(f"{name:<28}" + "".join(f"{metrics[m]:>16.3f}" for m in METRIC_ORDER))

    return 0


if __name__ == "__main__":
    sys.exit(main())
