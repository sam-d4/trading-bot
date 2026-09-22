"""Backtest 10 well-known published FOREX strategies (app/strategy/library.py) plus this
project's existing 3 rule-based baselines and buy-and-hold, across all 4 traded pairs, with the
SAME in-sample/out-of-sample split discipline as the RL validation gate (app/rl/evaluate.py):
every strategy uses fixed, textbook-standard parameters (no fitting on this data), is screened on
an in-sample window, and is only taken seriously if it ALSO holds up on a strictly held-out tail
it never touched - exactly the same reason app/rl/tournament.py backtests one strategy at a time
per random seed rather than trusting the single best in-sample result: with 14 candidates under
comparison, at least one will look good on any single window purely by chance even with zero true
edge (the standard multiple-comparisons trap in strategy research).

Usage:
    python scripts/backtest_strategy_library.py
    python scripts/backtest_strategy_library.py --instruments EUR_USD GBP_USD
"""

import argparse
import sys

import pandas as pd

from app.backtest.engine import buy_and_hold_result, run_backtest
from app.config.settings import get_settings
from app.data.enrichment import load_feature_context
from app.data.features import compute_features
from app.data.store import load_candles
from app.strategy.baselines import default_baseline_strategies
from app.strategy.library import add_extended_indicators, strategy_library

HOLDOUT_BARS = 500  # matches app/rl/scheduler.py's DEFAULT_HOLDOUT_BARS, for apples-to-apples comparison
GRANULARITY = "H1"


def _run_all(features: pd.DataFrame, strategies: list, label: str) -> dict[str, dict]:
    out = {}
    bh = buy_and_hold_result(features, granularity=GRANULARITY)
    out["buy_and_hold"] = bh.metrics
    for strat in strategies:
        result = run_backtest(features, strat, granularity=GRANULARITY)
        out[strat.name] = result.metrics
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"])
    args = parser.parse_args()

    settings = get_settings()
    strategies = [*default_baseline_strategies(), *strategy_library()]

    holdout_sharpe_by_strategy: dict[str, list[float]] = {}
    in_sample_sharpe_by_strategy: dict[str, list[float]] = {}

    for instrument in args.instruments:
        try:
            candles = load_candles(instrument, GRANULARITY)
        except FileNotFoundError as exc:
            print(f"[skip] {instrument}: {exc}")
            continue

        related, carry_differential = load_feature_context(settings, instrument, GRANULARITY)
        features = compute_features(candles, related_candles=related, carry_differential=carry_differential)
        features = add_extended_indicators(features)

        train = features.iloc[:-HOLDOUT_BARS].reset_index(drop=True)
        holdout = features.iloc[-HOLDOUT_BARS:].reset_index(drop=True)

        print(f"\n{'=' * 100}\n{instrument}  ({len(train)} in-sample bars, {len(holdout)} holdout bars)\n{'=' * 100}")

        in_sample_results = _run_all(train, strategies, "in-sample")
        holdout_results = _run_all(holdout, strategies, "holdout")

        header = f"{'strategy':<30}{'in-sample sharpe':>18}{'holdout sharpe':>18}{'holdout return%':>18}{'holdout trades':>16}"
        print(header)
        print("-" * len(header))
        for name in in_sample_results:
            in_s = in_sample_results[name]["sharpe"]
            hold = holdout_results[name]
            print(
                f"{name:<30}{in_s:>18.3f}{hold['sharpe']:>18.3f}"
                f"{hold['total_return_pct']:>18.3f}{hold['trade_count']:>16}"
            )
            if name != "buy_and_hold":
                holdout_sharpe_by_strategy.setdefault(name, []).append(hold["sharpe"])
                in_sample_sharpe_by_strategy.setdefault(name, []).append(in_s)

    print(f"\n{'=' * 100}\nAGGREGATE ACROSS ALL PAIRS (mean Sharpe) - the real comparison, not any single pair\n{'=' * 100}")
    header = f"{'strategy':<30}{'mean in-sample sharpe':>24}{'mean holdout sharpe':>22}{'pairs':>8}"
    print(header)
    print("-" * len(header))
    ranked = sorted(holdout_sharpe_by_strategy.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
    for name, values in ranked:
        in_s_values = in_sample_sharpe_by_strategy[name]
        mean_holdout = sum(values) / len(values)
        mean_in_sample = sum(in_s_values) / len(in_s_values)
        print(f"{name:<30}{mean_in_sample:>24.3f}{mean_holdout:>22.3f}{len(values):>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
