"""The RL self-improvement validation gate (plan Phase 7/8).

Backtests a freshly (re)trained candidate over a strictly held-out out-of-sample window - data
the candidate never trained on - and requires it to beat buy-and-hold, the best of the baseline
strategy library, AND the currently-live model (when one exists) before it can pass. A candidate
that fails is rejected and the live model is left untouched; only a pass makes it eligible for
promotion (still subject to the manual-confirmation setting - see app/rl/registry.py).
"""

from dataclasses import dataclass, field

import pandas as pd

from app.backtest.engine import buy_and_hold_result, run_backtest
from app.strategy.base import Strategy
from app.strategy.baselines import default_baseline_strategies


@dataclass
class ValidationThresholds:
    min_sharpe: float = 0.0
    max_allowed_drawdown_pct: float = 15.0
    min_trade_count: int = 10
    must_beat_buy_and_hold: bool = True
    must_beat_best_baseline: bool = True
    must_beat_live_model: bool = True


@dataclass
class ValidationOutcome:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    candidate_metrics: dict = field(default_factory=dict)
    comparison_metrics: dict = field(default_factory=dict)


def validate_candidate(
    candidate_strategy: Strategy,
    out_of_sample_features: pd.DataFrame,
    *,
    live_model_strategy: Strategy | None = None,
    granularity: str = "H1",
    thresholds: ValidationThresholds | None = None,
) -> ValidationOutcome:
    thresholds = thresholds or ValidationThresholds()
    reasons: list[str] = []

    candidate_result = run_backtest(out_of_sample_features, candidate_strategy, granularity=granularity)
    metrics = candidate_result.metrics

    if metrics["trade_count"] < thresholds.min_trade_count:
        reasons.append(
            f"too few trades ({metrics['trade_count']} < {thresholds.min_trade_count}) to be meaningful"
        )
    if metrics["sharpe"] < thresholds.min_sharpe:
        reasons.append(f"sharpe {metrics['sharpe']:.3f} below minimum {thresholds.min_sharpe}")
    if metrics["max_drawdown_pct"] < -thresholds.max_allowed_drawdown_pct:
        reasons.append(
            f"max drawdown {metrics['max_drawdown_pct']:.2f}% exceeds allowed "
            f"{-thresholds.max_allowed_drawdown_pct:.2f}%"
        )

    bh_result = buy_and_hold_result(out_of_sample_features, granularity=granularity)
    if thresholds.must_beat_buy_and_hold and metrics["sharpe"] <= bh_result.metrics["sharpe"]:
        reasons.append("did not beat buy-and-hold on Sharpe")

    baseline_results = {
        s.name: run_backtest(out_of_sample_features, s, granularity=granularity).metrics
        for s in default_baseline_strategies()
    }
    best_baseline_name = max(baseline_results, key=lambda n: baseline_results[n]["sharpe"])
    best_baseline_metrics = baseline_results[best_baseline_name]
    if thresholds.must_beat_best_baseline and metrics["sharpe"] <= best_baseline_metrics["sharpe"]:
        reasons.append(f"did not beat best baseline strategy ({best_baseline_name}) on Sharpe")

    comparison = {
        "buy_and_hold": bh_result.metrics,
        "best_baseline": {"name": best_baseline_name, **best_baseline_metrics},
        "live_model": None,
    }

    if live_model_strategy is not None:
        live_result = run_backtest(out_of_sample_features, live_model_strategy, granularity=granularity)
        comparison["live_model"] = live_result.metrics
        if thresholds.must_beat_live_model and metrics["sharpe"] <= live_result.metrics["sharpe"]:
            reasons.append("did not beat the currently-live model on Sharpe")

    return ValidationOutcome(
        passed=len(reasons) == 0,
        reasons=reasons,
        candidate_metrics=metrics,
        comparison_metrics=comparison,
    )
