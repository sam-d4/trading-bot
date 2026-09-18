"""Competitive tournament model selection.

Trains N independently-seeded candidates, evaluates every one of them on the SAME held-out
window starting from the same virtual account balance, and ranks them purely by ending account
value - direct competition, not an averaged metric. Only the single winner proceeds to the
existing validation gate (app/rl/evaluate.py) for the real "does this actually deserve to trade
real money" check; every other contestant is recorded as eliminated, with its rank and final
balance, for audit visibility in the model history.

This exists as a direct response to RL training's seed-sensitivity: a single training run's
outcome (e.g. converging to 0 trades) is not necessarily representative of what the same reward/
feature setup can achieve with different random initialization - see the run history in
scripts/run_training.py, where identical settings produced very different outcomes run to run.
Training several contestants and keeping only the best empirically-performing one makes
retraining robust to bad luck on any single run, without pretending the tournament itself is a
safety gate - it isn't; evaluate.py still is, applied only to the winner.
"""

import dataclasses

import structlog
from stable_baselines3 import PPO

from app.backtest.engine import run_backtest
from app.data.features import infer_feature_columns
from app.rl.pretrain import pretrain_policy
from app.rl.train import fine_tune, load_model
from app.strategy.base import Strategy
from app.strategy.rl_policy import RLPolicyStrategy

log = structlog.get_logger(__name__)


@dataclasses.dataclass
class Contestant:
    name: str
    seed: int
    ent_coef: float
    model: PPO
    strategy: RLPolicyStrategy
    final_balance: float
    metrics: dict
    rank: int = 0
    eliminated: bool = True


def run_tournament(
    train_features,
    holdout_features,
    teachers: list[Strategy],
    *,
    n_contestants: int = 4,
    lookback: int = 20,
    pretrain_epochs: int = 10,
    timesteps: int,
    ent_coef_range: tuple[float, float] = (0.005, 0.03),
    starting_balance: float = 10_000.0,
    granularity: str = "H1",
    base_seed: int = 0,
    name_prefix: str = "contestant",
    base_model_path: str | None = None,
    reward_mode: str | None = None,
    dsr_eta: float | None = None,
) -> tuple[Contestant, list[Contestant]]:
    """Returns (winner, all_contestants_ranked_best_first). winner.eliminated is always False and
    winner.rank == 1; every other entry has eliminated=True and rank 2..n_contestants.

    base_model_path: when given (the usual case once a live model exists - see
    app/rl/scheduler.py), every contestant starts as an independent copy of that model
    (reloaded fresh per contestant, never sharing one in-memory model - see the
    "two separate loads" note in scheduler.py for why that matters) and is re-seeded before
    fine-tuning, rather than cold-starting via pretrain_policy. This is what makes a *retrain*
    tournament ("N slightly different continuations of the live model, keep the best") distinct
    from the *cold-start* tournament (N independent behavior-cloning warm starts) used when
    there's no live model yet.
    """
    if n_contestants < 2:
        raise ValueError("a tournament needs at least 2 contestants to be meaningful")

    contestants: list[Contestant] = []
    for i in range(n_contestants):
        seed = base_seed + i
        # spread entropy coefficients evenly across the range so contestants explore differently
        # rather than all converging on the same degenerate solution together.
        ent_coef = ent_coef_range[0] + (ent_coef_range[1] - ent_coef_range[0]) * i / max(n_contestants - 1, 1)
        name = f"{name_prefix}_{i}_seed{seed}"

        log.info("tournament_contestant_training_start", index=i, name=name, seed=seed, ent_coef=round(ent_coef, 4))
        if base_model_path is not None:
            model = load_model(base_model_path)
            model.set_random_seed(seed)
            model.ent_coef = ent_coef
        else:
            model = pretrain_policy(
                train_features,
                teachers,
                lookback=lookback,
                epochs=pretrain_epochs,
                ent_coef=ent_coef,
                seed=seed,
                reward_mode=reward_mode,
                dsr_eta=dsr_eta,
            )
        model = fine_tune(
            model, train_features, total_timesteps=timesteps, lookback=lookback, reward_mode=reward_mode, dsr_eta=dsr_eta
        )

        strategy = RLPolicyStrategy(
            model, infer_feature_columns(train_features), lookback=lookback, model_version_name=name
        )
        result = run_backtest(holdout_features, strategy, granularity=granularity)
        final_balance = starting_balance * (1 + result.metrics["total_return_pct"] / 100)

        log.info(
            "tournament_contestant_result",
            name=name,
            final_balance=round(final_balance, 2),
            sharpe=result.metrics["sharpe"],
            trades=result.metrics["trade_count"],
        )
        contestants.append(
            Contestant(
                name=name,
                seed=seed,
                ent_coef=ent_coef,
                model=model,
                strategy=strategy,
                final_balance=final_balance,
                metrics=result.metrics,
            )
        )

    contestants.sort(key=lambda c: c.final_balance, reverse=True)
    for rank, c in enumerate(contestants, start=1):
        c.rank = rank
        c.eliminated = rank != 1

    winner = contestants[0]
    log.info(
        "tournament_complete",
        winner=winner.name,
        winner_balance=round(winner.final_balance, 2),
        starting_balance=starting_balance,
        eliminated=[(c.name, round(c.final_balance, 2)) for c in contestants[1:]],
    )
    return winner, contestants
