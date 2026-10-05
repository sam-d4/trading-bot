"""The full self-improvement cycle, run on a schedule per instrument (plan: tightened from
weekly to a short, configurable cadence - "fast" iteration, but every cycle still passes through
the same validation gate before it can touch live trading). One call to run_retrain_cycle does,
for ONE instrument:

  1. load cached historical data, split off a strictly held-out out-of-sample tail
  2. run a tournament of independently-seeded contestants (app/rl/tournament.py) - each
     continuing from that instrument's current live model's weights if one exists, or cold-start
     behavior-cloning from the baseline strategy library on its very first run - and rank them
     purely by ending virtual account balance on the held-out window
  3. only the tournament winner proceeds to the real validation gate (evaluate.py): compared
     against buy-and-hold, the baseline library, and that instrument's live model
  4. record every contestant's outcome (registry.py) - the winner is reject / queue-for-
     confirmation / auto-promote per the validation gate; every eliminated contestant is recorded
     rejected with its tournament rank, for audit visibility in the model history

Scheduling: one job per traded instrument, each on an interval of
interval_hours * len(traded_instruments), with staggered start times spaced interval_hours apart
- e.g. 4 instruments at interval_hours=1 means each individual pair retrains every 4 hours, but
the four jobs are offset so only one retrain (one tournament's worth of PPO training) is ever
running at a time, rather than quadrupling the background CPU load every cycle. Retraining every
pair "more often" would mean either accepting that higher constant load or shortening
interval_hours - a deliberate tradeoff, not an oversight. (4.0 -> 2.0 -> 1.0 in late Sep to speed
things up; back to 4.0 on 2026-10-05: replaying 277 models on fresh data showed more candidates per
day just meant more noise promotions, and the constant tournament training starved the live process
of CPU.)
"""

import datetime as dt

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config.settings import Settings
from app.data.enrichment import load_feature_context
from app.data.features import compute_features, infer_feature_columns
from app.data.features import INSTRUMENT_UNIVERSE
from app.data.store import load_candles, refresh_cached_candles
from app.notifications.alerts import AlertSender
from app.persistence import repo
from app.persistence.db import SessionLocal
from app.persistence.models import ModelStatus
from app.rl import registry, train
from app.rl.evaluate import ValidationThresholds, validate_candidate
from app.rl.tournament import run_tournament
from app.strategy.baselines import default_baseline_strategies
from app.strategy.rl_policy import RLPolicyStrategy

log = structlog.get_logger(__name__)

DEFAULT_LOOKBACK = 20
DEFAULT_TOTAL_TIMESTEPS = 20_000
# The held-out tail is split into two DISJOINT windows, both strictly after the training data:
# the tournament ranks contestants on the older one, and only the winner is validated on the newer
# one. Before, one 500-bar window did both jobs - the winner was picked on a window and then
# "validated" on that same window - and on 277 models replayed over fresh data the gate's score had a
# NEGATIVE rank correlation (-0.5) with real out-of-sample return (see scripts/evaluate_models_oos.py).
DEFAULT_SELECT_BARS = 750
DEFAULT_VALIDATE_BARS = 750
DEFAULT_CONTESTANTS = 3  # smaller than the manual script's default - this runs in the background


def run_retrain_cycle(
    settings: Settings,
    alerts: AlertSender,
    instrument: str,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    select_bars: int = DEFAULT_SELECT_BARS,
    validate_bars: int = DEFAULT_VALIDATE_BARS,
    granularity: str = "H1",
    n_contestants: int = DEFAULT_CONTESTANTS,
) -> None:
    # Bring the candle cache up to now first - it used to be refreshed by hand only, so every
    # candidate was trained and judged on data weeks old (ending Sep 11 when this was found Oct 5).
    for inst in INSTRUMENT_UNIVERSE:
        try:
            refresh_cached_candles(settings, inst, granularity)
        except Exception as exc:  # noqa: BLE001 - a flaky OANDA call must not kill the cycle; stale data beats none
            log.warning("candle_refresh_failed_using_cached_data", instrument=inst, error=str(exc)[:200])

    try:
        candles = load_candles(instrument, granularity)
    except FileNotFoundError:
        log.warning("retrain_skipped_no_cached_data", instrument=instrument)
        return

    related, carry_differential = load_feature_context(settings, instrument, granularity)
    features = compute_features(candles, related_candles=related, carry_differential=carry_differential)
    holdout_bars = select_bars + validate_bars
    min_rows = holdout_bars + lookback * 5
    if len(features) < min_rows:
        log.warning("retrain_skipped_insufficient_data", instrument=instrument, rows=len(features), required=min_rows)
        return

    train_features = features.iloc[:-holdout_bars].reset_index(drop=True)
    select_features = features.iloc[-holdout_bars:-validate_bars].reset_index(drop=True)
    validate_features = features.iloc[-validate_bars:].reset_index(drop=True)
    feature_columns = infer_feature_columns(train_features)

    with SessionLocal() as session:
        live_mv = repo.current_live_model(session, instrument)

    live_strategy = None
    base_model_path = None
    if live_mv is not None and live_mv.artifact_path:
        log.info("retrain_continuing_from_live_model", instrument=instrument, name=live_mv.name)
        base_model_path = live_mv.artifact_path
        # A fresh load, separate from whatever tournament.py loads for each contestant - the
        # comparison baseline must keep evaluating the live model's original (untouched) weights.
        live_strategy = RLPolicyStrategy(
            train.load_model(live_mv.artifact_path), feature_columns, lookback=lookback, model_version_name=live_mv.name
        )  # evaluated on validate_features only, so a fresh instance with an empty buffer is right
    else:
        log.info("retrain_cold_start_tournament_from_baselines", instrument=instrument)

    name_prefix = f"rl_{instrument}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    winner, contestants = run_tournament(
        train_features,
        select_features,
        default_baseline_strategies(),
        n_contestants=n_contestants,
        lookback=lookback,
        timesteps=total_timesteps,
        granularity=granularity,
        name_prefix=name_prefix,
        base_model_path=base_model_path,
    )

    with SessionLocal() as session:
        for c in contestants:
            artifact_path = train.save_model(c.model, settings.models_dir, c.name)
            mv = registry.register_candidate(
                session,
                name=c.name,
                instrument=instrument,
                lookback=lookback,
                artifact_path=artifact_path,
                training_data_start=train_features.iloc[0]["time"].to_pydatetime(),
                training_data_end=train_features.iloc[-1]["time"].to_pydatetime(),
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
                            f"balance (vs winner {winner.name})"
                        ],
                    },
                )
            else:
                winner_mv_id = mv.id

        # The winner was ranked on select_features; the gate must judge it on data it was NOT picked on.
        winner.strategy.reset()
        outcome = validate_candidate(
            winner.strategy,
            validate_features,
            live_model_strategy=live_strategy,
            granularity=granularity,
            thresholds=ValidationThresholds(),
        )
        registry.apply_validation_outcome(session, settings, winner_mv_id, outcome)

    if outcome.passed:
        log.info("retrain_cycle_candidate_validated", instrument=instrument, name=winner.name, metrics=outcome.candidate_metrics)
        if settings.require_manual_confirmation_to_promote:
            alerts.send(
                f"New {instrument} model candidate ready for review",
                f"{winner.name} (tournament winner of {len(contestants)})\n"
                f"sharpe={outcome.candidate_metrics['sharpe']:.3f} "
                f"max_dd={outcome.candidate_metrics['max_drawdown_pct']:.2f}% "
                f"trades={outcome.candidate_metrics['trade_count']}",
            )
    else:
        log.info("retrain_cycle_candidate_rejected", instrument=instrument, name=winner.name, reasons=outcome.reasons)


def start_retrain_scheduler(
    settings: Settings, alerts: AlertSender, *, interval_hours: float = 4.0
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    instruments = settings.traded_instruments
    per_instrument_interval = interval_hours * len(instruments)

    for i, instrument in enumerate(instruments):
        scheduler.add_job(
            run_retrain_cycle,
            "interval",
            hours=per_instrument_interval,
            args=[settings, alerts, instrument],
            next_run_time=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5) + dt.timedelta(hours=interval_hours * i),
            id=f"rl_retrain_cycle_{instrument}",
            max_instances=1,
        )
    scheduler.start()
    return scheduler
