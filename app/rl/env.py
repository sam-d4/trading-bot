"""The RL training environment. A single-instrument, single-position trading env over a fixed
historical feature window, following the plan's design: the agent picks DIRECTION ONLY
(flat/long/short via Discrete(3), matching app/strategy/base.py's Signal enum) - sizing and risk
are never the RL model's concern, so a policy bug can never translate into an oversized bet
regardless of how this environment is trained or evaluated.

Two reward modes (`reward_mode`):

  "pnl" (the original design): raw net-of-cost per-step return, matching
  app/backtest/engine.py's accounting exactly. Simple and directly interpretable, but creates a
  real mismatch: the validation gate (app/rl/evaluate.py) judges candidates on Sharpe ratio, not
  raw return, so a policy trained to maximize P&L isn't actually optimizing what it's graded on.
  In practice (see scripts/run_training.py's run history) this mode found no consistent edge
  across several tournament rounds - not necessarily because none exists, but because raw-P&L
  reward gives no special credit for *risk-adjusted* quality, so a lucky high-variance policy and
  a genuinely-better low-variance one can score identically in training while scoring very
  differently on the Sharpe-based gate they're actually judged against at the end.

  "differential_sharpe" (the default): the Differential Sharpe Ratio from Moody & Saffell,
  "Learning to Trade via Direct Reinforcement" (1998) - a per-step reward whose sum over an
  episode telescopes to (approximately) the episode's Sharpe ratio, computed from running
  exponential-moving-average estimates of the first and second moments of returns. This directly
  aligns the training objective with the evaluation objective, and inherently penalizes
  volatility rather than only rewarding raw return - a fundamentally different shape of reward
  from "pnl", not just a retuned constant.

Both modes still subtract DEFAULT_FLAT_PENALTY_BPS's tiny standing cost for staying flat (see
that constant's docstring for why - it fixes a real, previously-observed "collapse to always-
flat" failure mode that's a separate problem from which reward mode is in use). In both modes,
app/backtest/engine.py's run_backtest (used for the validation gate and everywhere downstream of
training) always scores true P&L net of real costs only - none of this reward shaping leaks into
how a candidate is actually judged.
"""

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from app.data.features import infer_feature_columns
from app.strategy.base import Signal

ACTION_TO_POSITION = {0: 0, 1: 1, 2: -1}  # matches Signal.FLAT/LONG/SHORT ordering
TRANSACTION_COST_BPS = 1.0
# A per-step cost for sitting flat - deliberately smaller than TRANSACTION_COST_BPS (1.0) so it
# never makes a bad trade look good, but large enough to meaningfully outweigh a typical hourly
# log-return. Its job is to break "always flat" as a stable zero-reward equilibrium: with pure
# P&L reward, once PPO discovers most trades lose to costs, staying flat forever scores exactly 0
# and nothing in the reward signal ever pushes it to reconsider (observed repeatedly in practice -
# see scripts/run_training.py's training history: 400k-timestep runs converging to 0 trades).
# Raised from 0.1 to 0.3 (2026-09-19) to push future tournament winners toward trading more
# often - live models were validated but signaling only every 2-5 days per pair, which is too
# slow to accumulate the trade history the self-improvement loop needs to learn from. Still an
# order of magnitude below the transaction cost, so it nudges frequency without paying for
# obviously-losing churn.
DEFAULT_FLAT_PENALTY_BPS = 0.3
DEFAULT_REWARD_MODE = "differential_sharpe"
DEFAULT_DSR_ETA = 0.02  # EMA decay for the running return-moment estimates; ~50-step effective window
DSR_VARIANCE_FLOOR = 1e-8  # guards the Sharpe denominator before enough steps have built up real variance


class TradingEnv(Env):
    """features_df: output of app.data.features.compute_features. lookback: number of trailing
    feature rows stacked into each observation, giving the policy short-term context without
    needing a recurrent architecture."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        *,
        lookback: int = 20,
        transaction_cost_bps: float = TRANSACTION_COST_BPS,
        flat_penalty_bps: float = DEFAULT_FLAT_PENALTY_BPS,
        reward_mode: str = DEFAULT_REWARD_MODE,
        dsr_eta: float = DEFAULT_DSR_ETA,
    ):
        super().__init__()
        if len(features_df) <= lookback + 1:
            raise ValueError(f"features_df too short ({len(features_df)}) for lookback={lookback}")
        if reward_mode not in ("pnl", "differential_sharpe"):
            raise ValueError(f"unknown reward_mode {reward_mode!r} - expected 'pnl' or 'differential_sharpe'")

        self._df = features_df.reset_index(drop=True)
        self.feature_columns = infer_feature_columns(self._df)
        self._feature_matrix = self._df[self.feature_columns].to_numpy(dtype=np.float32)
        self._log_returns = self._df["log_return"].to_numpy(dtype=np.float32)
        self._lookback = lookback
        self._transaction_cost = transaction_cost_bps / 10_000
        self._flat_penalty = flat_penalty_bps / 10_000
        self._reward_mode = reward_mode
        self._dsr_eta = dsr_eta

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback * len(self.feature_columns) + 1,), dtype=np.float32
        )

        self._position = 0
        self._step_idx = 0
        self._dsr_mean = 0.0  # running EMA of net return ("A" in Moody & Saffell's notation)
        self._dsr_mean_sq = 0.0  # running EMA of squared net return ("B")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._position = 0
        self._step_idx = self._lookback
        self._dsr_mean = 0.0
        self._dsr_mean_sq = 0.0
        return self._observation(), {}

    def step(self, action: int):
        new_position = ACTION_TO_POSITION[int(action)]
        cost = abs(new_position - self._position) * self._transaction_cost
        flat_cost = self._flat_penalty if new_position == 0 else 0.0

        # the position taken at this step earns the NEXT bar's return - same one-bar lag as
        # app/backtest/engine.py, so live/backtest/train all agree on what "acting on bar t" means.
        next_idx = self._step_idx + 1
        gross_return = new_position * self._log_returns[next_idx] if next_idx < len(self._log_returns) else 0.0
        net_return = gross_return - cost - flat_cost

        reward = self._differential_sharpe(net_return) if self._reward_mode == "differential_sharpe" else net_return

        self._position = new_position
        self._step_idx += 1

        terminated = self._step_idx >= len(self._df) - 1
        truncated = False
        obs = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"position": self._position, "net_return": net_return}
        return obs, float(reward), terminated, truncated, info

    def _differential_sharpe(self, net_return: float) -> float:
        """Moody & Saffell's differential Sharpe ratio: the reward whose per-step sum tracks the
        Sharpe ratio's evolution, rather than raw return. Uses THIS step's return against the
        PRIOR step's running moments (that ordering, not the updated ones, is what makes it a
        true differential/derivative rather than just a rescaled return), then updates the
        running moments for next time."""
        prev_mean, prev_mean_sq = self._dsr_mean, self._dsr_mean_sq
        delta_mean = net_return - prev_mean
        delta_mean_sq = net_return**2 - prev_mean_sq

        variance = prev_mean_sq - prev_mean**2
        if variance <= DSR_VARIANCE_FLOOR:
            # not enough history yet to have a meaningful variance estimate (e.g. right after
            # reset, or a run of identical returns) - fall back to plain return rather than
            # dividing by ~0, which would produce an exploding, meaningless reward spike.
            dsr = net_return
        else:
            dsr = (prev_mean_sq * delta_mean - 0.5 * prev_mean * delta_mean_sq) / variance**1.5

        self._dsr_mean += self._dsr_eta * delta_mean
        self._dsr_mean_sq += self._dsr_eta * delta_mean_sq
        return dsr

    def _observation(self) -> np.ndarray:
        window = self._feature_matrix[self._step_idx - self._lookback : self._step_idx]
        return np.concatenate([window.flatten(), [float(self._position)]]).astype(np.float32)

    def baseline_action_for_signal(self, signal: Signal) -> int:
        return {Signal.FLAT: 0, Signal.LONG: 1, Signal.SHORT: 2}[signal]

    def current_features_row(self) -> pd.Series:
        return self._df.iloc[self._step_idx]
