"""End-to-end RL training run: a tournament of independently-seeded candidates competes on the
held-out window (ranked purely by ending virtual account balance - see app/rl/tournament.py),
the losers are eliminated and recorded, and only the winner proceeds to the real validation gate
(-> registered candidate/validated/rejected, never straight to LIVE unless
require_manual_confirmation_to_promote is off - see app/rl/registry.py).

This is what app/rl/scheduler.py runs automatically on its retrain cadence once historical data
exists; this script is the same pipeline run by hand for the first training pass, or for manual
experimentation.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --instrument EUR_USD --timesteps 100000 --contestants 4
"""

import argparse
import datetime as dt
import sys

from app.config.settings import get_settings
from app.data.enrichment import load_feature_context
from app.data.features import compute_features, infer_feature_columns
from app.data.store import load_candles
from app.persistence import repo
from app.persistence.db import SessionLocal, init_db
from app.persistence.models import ModelStatus
from app.rl.evaluate import validate_candidate
from app.rl.registry import apply_validation_outcome, register_candidate
from app.rl.tournament import run_tournament
from app.rl.train import save_model
from app.strategy.baselines import default_baseline_strategies
from app.strategy.rl_policy import RLPolicyStrategy

GRANULARITY_BARS_PER_WEEK = {"M5": 12 * 24 * 5, "M15": 4 * 24 * 5, "H1": 24 * 5, "H4": 6 * 5, "D": 5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default=None, help="defaults to PRIMARY_INSTRUMENT from .env")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--holdout-weeks", type=int, default=8)
    parser.add_argument(
        "--holdout-offset-weeks",
        type=int,
        default=0,
        help="shift the holdout window back this many weeks from the end of cached data (0 = most "
        "recent window, as before) - lets you test the same setup against an earlier, different "
        "market regime rather than always the same recent window. Training data is always "
        "everything strictly before the holdout window, never anything after it.",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=50_000, help="PPO timesteps PER contestant")
    parser.add_argument("--contestants", type=int, default=4, help="how many independently-seeded candidates compete")
    parser.add_argument(
        "--ent-coef-min", type=float, default=0.005, help="lowest PPO entropy bonus across contestants"
    )
    parser.add_argument(
        "--ent-coef-max", type=float, default=0.03, help="highest PPO entropy bonus across contestants"
    )
    parser.add_argument(
        "--reward-mode",
        choices=["pnl", "differential_sharpe"],
        default=None,
        help="training reward shape - defaults to the env's own default (differential_sharpe)",
    )
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    instrument = args.instrument or settings.primary_instrument

    try:
        candles = load_candles(instrument, args.granularity)
    except FileNotFoundError as exc:
        print(exc)
        print(f"Run: python scripts/fetch_historical_data.py --instruments {instrument} --granularity {args.granularity}")
        return 1

    related, carry_differential = load_feature_context(settings, instrument, args.granularity)
    print(f"Cross-pair data: {list(related.keys()) or 'none cached'}, carry_differential={carry_differential:.4f}")
    features = compute_features(candles, related_candles=related, carry_differential=carry_differential)
    bars_per_week = GRANULARITY_BARS_PER_WEEK.get(args.granularity, 24 * 5)
    holdout_bars = args.holdout_weeks * bars_per_week
    offset_bars = args.holdout_offset_weeks * bars_per_week
    holdout_end = len(features) - offset_bars
    holdout_start = holdout_end - holdout_bars
    if holdout_start <= len(features) // 4:  # keep a sane minimum amount of training history
        print(
            f"Not enough cached data for an {args.holdout_weeks}-week holdout "
            f"{f'offset {args.holdout_offset_weeks} weeks back ' if args.holdout_offset_weeks else ''}"
            f"({len(features)} rows total)."
        )
        return 1

    # Training data is only ever what's strictly BEFORE the holdout window - even when the
    # window is offset back in time, data chronologically after it is left unused rather than
    # trained on, so this stays a genuine "train on the past, test on this window" split, never
    # accidentally training on data from after the period being tested.
    train_features = features.iloc[:holdout_start].reset_index(drop=True)
    holdout_features = features.iloc[holdout_start:holdout_end].reset_index(drop=True)
    print(
        f"{len(train_features)} training rows "
        f"({train_features['time'].iloc[0].date()} -> {train_features['time'].iloc[-1].date()}), "
        f"{len(holdout_features)} holdout rows "
        f"({holdout_features['time'].iloc[0].date()} -> {holdout_features['time'].iloc[-1].date()})"
    )

    name_prefix = f"{instrument}_{args.granularity}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(
        f"\nTournament: {args.contestants} contestants, {args.timesteps} PPO timesteps each, "
        f"reward_mode={args.reward_mode or '(env default)'}, "
        f"starting from £10,000 virtual balance, competing on the held-out window...\n"
    )
    winner, contestants = run_tournament(
        train_features,
        holdout_features,
        default_baseline_strategies(),
        n_contestants=args.contestants,
        lookback=args.lookback,
        pretrain_epochs=args.pretrain_epochs,
        timesteps=args.timesteps,
        ent_coef_range=(args.ent_coef_min, args.ent_coef_max),
        reward_mode=args.reward_mode,
        granularity=args.granularity,
        name_prefix=name_prefix,
    )

    print("Tournament results (ranked by ending balance):")
    for c in contestants:
        marker = "WINNER" if not c.eliminated else "eliminated"
        print(
            f"  #{c.rank} {c.name:<45} balance=£{c.final_balance:>10,.2f}  "
            f"sharpe={c.metrics['sharpe']:>7.3f}  trades={c.metrics['trade_count']:>4}  [{marker}]"
        )

    with SessionLocal() as session:
        # Every contestant is registered for audit visibility, not just the winner - the
        # dashboard's model history should show the full tournament, not just who won.
        for c in contestants:
            artifact_path = save_model(c.model, settings.models_dir, c.name)
            mv = register_candidate(
                session,
                name=c.name,
                instrument=instrument,
                lookback=args.lookback,
                artifact_path=artifact_path,
                training_data_start=train_features["time"].iloc[0].to_pydatetime(),
                training_data_end=train_features["time"].iloc[-1].to_pydatetime(),
            )
            if c.eliminated:
                repo.set_model_status(
                    session,
                    mv.id,
                    ModelStatus.REJECTED,
                    validation_metrics={
                        "candidate": c.metrics,
                        "reasons": [
                            f"eliminated in tournament: ranked #{c.rank} of {len(contestants)} by ending "
                            f"balance (£{c.final_balance:,.2f} vs winner's £{winner.final_balance:,.2f})"
                        ],
                    },
                )
            else:
                winner_model_version_id = mv.id

        live_model = repo.current_live_model(session, instrument)
        live_strategy = None
        if live_model is not None and live_model.artifact_path:
            live_strategy = RLPolicyStrategy.load(
                live_model.artifact_path,
                infer_feature_columns(train_features),
                lookback=live_model.lookback or args.lookback,
                model_version_name=live_model.name,
            )

        print(f"\nValidating tournament winner ({winner.name}) against buy-and-hold, baselines, current live model...")
        outcome = validate_candidate(
            winner.strategy, holdout_features, live_model_strategy=live_strategy, granularity=args.granularity
        )
        updated = apply_validation_outcome(session, settings, winner_model_version_id, outcome)

    print(f"\nWinner's candidate metrics: {outcome.candidate_metrics}")
    print(f"Comparison metrics:         {outcome.comparison_metrics}")
    print(f"\nPassed validation gate: {outcome.passed}")
    if outcome.reasons:
        print("Reasons:")
        for reason in outcome.reasons:
            print(f"  - {reason}")
    print(f"\nModel '{winner.name}' status: {updated.status.value if updated else 'unknown'}")
    if updated and updated.status.value == "validated":
        print("Awaiting manual confirmation on the dashboard before it can go live.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
