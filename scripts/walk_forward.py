"""Walk-forward evaluation of the RL training recipe on one instrument.

For each of several points spread across the history: train N candidates FROM SCRATCH on everything
before a selection window, rank them on that selection window (what the tournament does), then test
every candidate - and the pick the pipeline would deploy, and a majority-vote ensemble - on the
period right AFTER, which nothing has seen. Averaged across folds this answers the questions a single
validation window can't: does the recipe make money across market regimes, does picking the
tournament winner beat picking at random, and does the vote beat a single model?

Usage:
    python scripts/walk_forward.py --instrument AUD_USD --folds 6 --candidates 6 --timesteps 20000
    python scripts/walk_forward.py --instrument EUR_USD --flat-penalty 0.1 --reward-mode pnl   # a different recipe
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from app.backtest.engine import buy_and_hold_result, run_backtest  # noqa: E402
from app.config.settings import get_settings  # noqa: E402
from app.data.enrichment import load_feature_context  # noqa: E402
from app.data.features import compute_features  # noqa: E402
from app.data.store import load_candles  # noqa: E402
from app.rl.tournament import run_tournament  # noqa: E402
from app.rl.walk_forward import VoteStrategy, fold_bounds, summarize_folds  # noqa: E402
from app.strategy.baselines import default_baseline_strategies  # noqa: E402

WARMUP = 25  # rows fed before the test window so a policy's lookback buffer is warm when scoring starts


def score(features: pd.DataFrame, start: int, end: int, strategy, cost_bps: float) -> dict:
    if hasattr(strategy, "reset"):
        strategy.reset()
    window = features.iloc[start - WARMUP : end].reset_index(drop=True)
    return run_backtest(window, strategy, transaction_cost_bps=cost_bps).metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--test-bars", type=int, default=750)
    parser.add_argument("--select-bars", type=int, default=750)
    parser.add_argument("--stride", type=int, default=3000, help="bars between consecutive folds' test windows")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--flat-penalty", type=float, default=None, help="bps; default = env.py's DEFAULT_FLAT_PENALTY_BPS")
    parser.add_argument("--reward-mode", default=None, choices=["pnl", "differential_sharpe"])
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (the live bot shares this machine)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    settings = get_settings()
    instrument = args.instrument
    out = Path(args.out or f"logs/walk_forward_{instrument}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    candles = load_candles(instrument, "H1")
    related, carry = load_feature_context(settings, instrument, "H1")
    features = compute_features(candles, related_candles=related, carry_differential=carry)
    folds = fold_bounds(
        len(features), folds=args.folds, test_bars=args.test_bars, select_bars=args.select_bars, stride=args.stride
    )
    print(f"{instrument}: {len(features)} bars, {len(folds)} folds, {args.candidates} candidates each, "
          f"{args.timesteps} timesteps, flat_penalty={args.flat_penalty}, reward={args.reward_mode}", flush=True)

    rows = []
    for fold in folds:
        t0 = time.time()
        train = features.iloc[: fold.train_end].reset_index(drop=True)
        select = features.iloc[fold.train_end : fold.select_end].reset_index(drop=True)
        label = f"{features['time'].iloc[fold.select_end].date()}..{features['time'].iloc[fold.test_end - 1].date()}"

        _, contestants = run_tournament(  # ranked best-first on the SELECT window
            train, select, default_baseline_strategies(),
            n_contestants=args.candidates, timesteps=args.timesteps, base_seed=fold.index * 100,
            name_prefix=f"wf_{instrument}_f{fold.index}", reward_mode=args.reward_mode,
            flat_penalty_bps=args.flat_penalty,
        )
        for c in contestants:
            m = score(features, fold.select_end, fold.test_end, c.strategy, args.cost_bps)
            rows.append(dict(fold=fold.index, window=label, kind="candidate", rank=c.rank,
                             select_ret=c.metrics["total_return_pct"], select_sharpe=c.metrics["sharpe"],
                             test_ret=m["total_return_pct"], test_sharpe=m["sharpe"], test_trades=m["trade_count"]))
        for name, members in (("vote_all", contestants), ("vote_top3", contestants[:3])):
            m = score(features, fold.select_end, fold.test_end, VoteStrategy([c.strategy for c in members], name), args.cost_bps)
            rows.append(dict(fold=fold.index, window=label, kind=name, test_ret=m["total_return_pct"],
                             test_sharpe=m["sharpe"], test_trades=m["trade_count"]))
        bh = buy_and_hold_result(features.iloc[fold.select_end : fold.test_end].reset_index(drop=True)).metrics
        rows.append(dict(fold=fold.index, window=label, kind="buy_and_hold", test_ret=bh["total_return_pct"]))
        for b in default_baseline_strategies():
            m = score(features, fold.select_end, fold.test_end, b, args.cost_bps)
            rows.append(dict(fold=fold.index, window=label, kind=b.name, test_ret=m["total_return_pct"], test_trades=m["trade_count"]))

        pd.DataFrame(rows).to_csv(out, index=False)
        fr = pd.DataFrame([r for r in rows if r["fold"] == fold.index])
        cand = fr[fr.kind == "candidate"]
        print(f"fold {fold.index} test {label} ({time.time() - t0:.0f}s) | candidates mean {cand.test_ret.mean():+.2f}% "
              f"| winner(rank1) {cand[cand['rank'] == 1].test_ret.iloc[0]:+.2f}% "
              f"| vote_all {fr[fr.kind == 'vote_all'].test_ret.iloc[0]:+.2f}% "
              f"| buy&hold {fr[fr.kind == 'buy_and_hold'].test_ret.iloc[0]:+.2f}%", flush=True)

    print(f"\n=== {instrument}: out-of-sample, cost {args.cost_bps}bp/side ===")
    print(summarize_folds(pd.DataFrame(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
