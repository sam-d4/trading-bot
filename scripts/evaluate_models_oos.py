"""Replay every registered RL model on an out-of-sample window with correct H1 inputs, and compare
the models the pipeline SELECTED (live / validated / retired) against the ones it REJECTED.

Why this exists: the validation gate picks winners on one short, reused window, so "selected" could
just mean "lucky on that window". If selected models do no better than rejected ones on data none of
them ever saw, the selection process has no predictive power and promoting on it is a coin flip.

Usage:
    python scripts/evaluate_models_oos.py --since 2026-09-11T14:00:00
    python scripts/evaluate_models_oos.py --since 2026-09-11T14:00:00 --cost-bps 2.0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app.backtest.engine import buy_and_hold_result, run_backtest
from app.config.settings import get_settings
from app.data.enrichment import load_feature_context
from app.data.features import compute_features, infer_feature_columns
from app.data.store import load_candles
from app.persistence.db import SessionLocal
from app.persistence.models import ModelVersion
from app.rl import train
from app.strategy.baselines import default_baseline_strategies
from app.strategy.rl_policy import RLPolicyStrategy

SELECTED = {"live", "validated", "retired"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="start of the out-of-sample window (UTC)")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"])
    parser.add_argument("--out", default=None, help="optional CSV path for the per-model results")
    args = parser.parse_args()

    settings = get_settings()
    since = pd.Timestamp(args.since, tz="UTC")
    rows = []

    for instrument in args.instruments:
        candles = load_candles(instrument, "H1")
        related, carry = load_feature_context(settings, instrument, "H1")
        features = compute_features(candles, related_candles=related, carry_differential=carry)
        cols = infer_feature_columns(features)

        oos_idx = features.index[features["time"] > since]
        if len(oos_idx) < 100:
            print(f"{instrument}: only {len(oos_idx)} bars after {since} - skipping")
            continue
        # start 25 rows early so the policy's lookback buffer is warm when scoring begins
        window = features.loc[max(oos_idx[0] - 25, 0):].reset_index(drop=True)
        bh = buy_and_hold_result(window.iloc[25:].reset_index(drop=True)).metrics
        print(f"\n{instrument}: {len(oos_idx)} out-of-sample bars ({features.loc[oos_idx[0], 'time']} -> {features['time'].iloc[-1]})"
              f" | buy&hold return {bh['total_return_pct']:+.2f}%")

        for strat in default_baseline_strategies():
            m = run_backtest(window, strat, transaction_cost_bps=args.cost_bps).metrics
            rows.append(dict(instrument=instrument, kind="baseline", name=strat.name, status="baseline", **m))

        with SessionLocal() as session:
            models = list(session.scalars(select(ModelVersion).where(
                ModelVersion.instrument == instrument, ModelVersion.artifact_path.isnot(None))))
        skipped = 0
        for mv in models:
            if mv.training_data_end is not None and pd.Timestamp(mv.training_data_end, tz="UTC") > since:
                continue  # trained on data inside the window - not out-of-sample
            if not Path(mv.artifact_path).exists():
                skipped += 1
                continue
            lookback = mv.lookback or 20
            model = train.load_model(mv.artifact_path)
            if model.observation_space.shape[0] != lookback * len(cols) + 1:
                skipped += 1  # trained on a different feature set (older pipeline version)
                continue
            strat = RLPolicyStrategy(model, cols, lookback=lookback, model_version_name=mv.name)
            m = run_backtest(window, strat, transaction_cost_bps=args.cost_bps).metrics
            rows.append(dict(instrument=instrument, kind="rl", name=mv.name, status=mv.status.value, **m))
        print(f"  evaluated {sum(r['instrument'] == instrument and r['kind'] == 'rl' for r in rows)} RL models ({skipped} skipped: missing file / old feature set)")

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)

    rl = df[df.kind == "rl"].copy()
    rl["group"] = rl["status"].map(lambda s: "selected (live/validated/retired)" if s in SELECTED else "rejected")
    pd.set_option("display.width", 200)
    print("\n=== RL models on out-of-sample data, by what the pipeline decided ===")
    print(rl.groupby(["instrument", "group"]).agg(
        models=("name", "count"),
        mean_return_pct=("total_return_pct", "mean"),
        median_return_pct=("total_return_pct", "median"),
        mean_sharpe=("sharpe", "mean"),
        share_profitable=("total_return_pct", lambda x: (x > 0).mean()),
        mean_trades=("trade_count", "mean"),
    ).round(3).to_string())
    print("\n=== all pairs together ===")
    print(rl.groupby("group").agg(
        models=("name", "count"),
        mean_return_pct=("total_return_pct", "mean"),
        median_return_pct=("total_return_pct", "median"),
        share_profitable=("total_return_pct", lambda x: (x > 0).mean()),
        mean_trades=("trade_count", "mean"),
    ).round(3).to_string())
    print("\n=== baselines ===")
    print(df[df.kind == "baseline"][["instrument", "name", "total_return_pct", "sharpe", "trade_count"]].round(3).to_string(index=False))
    print("\n=== currently/previously LIVE models ===")
    print(rl[rl.status.isin(["live", "retired"])][["instrument", "name", "status", "total_return_pct", "sharpe", "trade_count"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
