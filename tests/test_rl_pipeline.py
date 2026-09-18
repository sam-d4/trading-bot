"""End-to-end smoke test of the RL pipeline: behavior-cloning warm start -> PPO fine-tuning ->
validation gate. Uses tiny synthetic data and a handful of timesteps purely to prove the pipeline
runs without error and produces sane types/shapes - this is NOT a claim that the resulting policy
is any good (that requires real historical data and much longer training, per Phase 7/8 of the
plan). Marked slow since it drives actual PyTorch training, not just pure-Python logic.
"""

import pytest

from app.data.features import infer_feature_columns
from app.rl.evaluate import ValidationThresholds, validate_candidate
from app.rl.pretrain import pretrain_policy
from app.rl.train import fine_tune
from app.strategy.baselines import default_baseline_strategies
from app.strategy.rl_policy import RLPolicyStrategy
from tests.synthetic_data import synthetic_features

pytestmark = pytest.mark.slow


def test_pretrain_finetune_validate_pipeline_runs_end_to_end():
    features = synthetic_features()
    train_features = features.iloc[:-200].reset_index(drop=True)
    holdout_features = features.iloc[-200:].reset_index(drop=True)

    model = pretrain_policy(train_features, default_baseline_strategies(), lookback=10, epochs=2)
    model = fine_tune(model, train_features, total_timesteps=256, lookback=10)

    candidate = RLPolicyStrategy(model, infer_feature_columns(train_features), lookback=10, model_version_name="pipeline_test")
    outcome = validate_candidate(
        candidate, holdout_features, thresholds=ValidationThresholds(min_trade_count=0)
    )

    assert isinstance(outcome.passed, bool)
    assert "sharpe" in outcome.candidate_metrics
    assert "buy_and_hold" in outcome.comparison_metrics
    assert "best_baseline" in outcome.comparison_metrics
