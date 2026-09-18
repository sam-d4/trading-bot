"""Smoke test for the competitive tournament training path (app/rl/tournament.py). Tiny
synthetic data and a handful of timesteps, purely to prove the pipeline runs and produces
correctly-ranked, correctly-flagged contestants - not a claim about policy quality (see
test_rl_pipeline.py's same caveat). Marked slow since it drives actual PyTorch training.
"""

import pytest

from app.rl.tournament import run_tournament
from app.rl.train import save_model
from app.strategy.baselines import default_baseline_strategies
from tests.synthetic_data import synthetic_features

pytestmark = pytest.mark.slow


def test_tournament_ranks_contestants_and_flags_exactly_one_winner():
    features = synthetic_features(n=500, seed=5)
    train_features = features.iloc[:-150].reset_index(drop=True)
    holdout_features = features.iloc[-150:].reset_index(drop=True)

    winner, contestants = run_tournament(
        train_features,
        holdout_features,
        default_baseline_strategies(),
        n_contestants=3,
        lookback=10,
        pretrain_epochs=2,
        timesteps=128,
    )

    assert len(contestants) == 3
    assert winner is contestants[0]
    assert winner.eliminated is False
    assert winner.rank == 1
    assert all(c.eliminated for c in contestants[1:])
    assert [c.rank for c in contestants] == [1, 2, 3]
    # ranked strictly by descending final balance
    balances = [c.final_balance for c in contestants]
    assert balances == sorted(balances, reverse=True)


def test_tournament_continuation_mode_starts_from_base_model(tmp_path):
    features = synthetic_features(n=500, seed=6)
    train_features = features.iloc[:-150].reset_index(drop=True)
    holdout_features = features.iloc[-150:].reset_index(drop=True)

    # Produce a base model the same way a cold-start tournament would, then use it as the
    # continuation base - mirrors app/rl/scheduler.py's "continue from the live model" path.
    seed_winner, _ = run_tournament(
        train_features, holdout_features, default_baseline_strategies(),
        n_contestants=2, lookback=10, pretrain_epochs=2, timesteps=64,
    )
    base_path = save_model(seed_winner.model, tmp_path, "base_model")

    winner, contestants = run_tournament(
        train_features,
        holdout_features,
        default_baseline_strategies(),
        n_contestants=2,
        lookback=10,
        timesteps=64,
        base_model_path=base_path,
    )

    assert len(contestants) == 2
    assert winner.eliminated is False
    assert contestants[1].eliminated is True


def test_tournament_requires_at_least_two_contestants():
    features = synthetic_features(n=300)
    train_features = features.iloc[:-100].reset_index(drop=True)
    holdout_features = features.iloc[-100:].reset_index(drop=True)

    with pytest.raises(ValueError):
        run_tournament(
            train_features, holdout_features, default_baseline_strategies(),
            n_contestants=1, timesteps=64,
        )
