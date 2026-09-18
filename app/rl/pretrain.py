"""Behavior-cloning warm start: fits a fresh PPO policy's action distribution to the baseline
strategy library's decisions on historical data, via supervised learning, BEFORE any RL training
happens. This is the mechanism behind the plan's "cold-start knowledge" approach - day-one policy
weights already encode "a reasonable rule-based trader" instead of being random, so
train.py's PPO fine-tuning is improving on a real starting point rather than discovering trading
from nothing.

Standard technique for warm-starting an SB3 policy: there's no built-in `pretrain()` in modern
SB3 (removed after v1), so this drives the policy network directly via its own
`evaluate_actions` to get action log-probabilities, and minimizes their negative log-likelihood
against the teacher's labels - ordinary supervised classification, just implemented against
SB3's policy object instead of a plain torch classifier.
"""

import numpy as np
import structlog
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from app.rl.env import TradingEnv
from app.strategy.base import Strategy

log = structlog.get_logger(__name__)


def build_labeled_dataset(env: TradingEnv, teacher: Strategy) -> tuple[np.ndarray, np.ndarray]:
    """Replays the environment start-to-end under a rule-based teacher, recording
    (observation, teacher_action) pairs - the supervised dataset for behavior cloning."""
    obs, _ = env.reset()
    observations = []
    actions = []
    terminated = False
    while not terminated:
        row = env.current_features_row()
        signal = teacher.decide(row)
        action = env.baseline_action_for_signal(signal)
        observations.append(obs)
        actions.append(action)
        obs, _, terminated, _, _ = env.step(action)
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def behavior_clone(
    policy: BasePolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> BasePolicy:
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    device = policy.device

    obs_tensor_all = torch.as_tensor(observations, device=device)
    actions_tensor_all = torch.as_tensor(actions, device=device)
    n = len(observations)
    if n == 0:
        raise ValueError("no training examples provided to behavior_clone")

    for epoch in range(epochs):
        permutation = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            obs_batch = obs_tensor_all[idx]
            action_batch = actions_tensor_all[idx]

            _, log_prob, _ = policy.evaluate_actions(obs_batch, action_batch)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        log.info("behavior_cloning_epoch", epoch=epoch, avg_loss=total_loss / n)

    return policy


def pretrain_policy(
    features_df,
    teachers: list[Strategy],
    *,
    lookback: int = 20,
    epochs: int = 10,
    ent_coef: float = 0.01,
    flat_penalty_bps: float | None = None,
    reward_mode: str | None = None,
    dsr_eta: float | None = None,
    seed: int | None = None,
) -> PPO:
    """Builds a fresh PPO model over a TradingEnv sized to features_df, then warm-starts its
    policy by imitating every strategy in `teachers` (datasets concatenated - the policy learns
    to imitate a mix of reasonable trading styles, not just one). Returns the PPO model, ready
    for train.py to continue with ordinary RL fine-tuning (model.learn(...)).

    ent_coef defaults to SB3's stock 0.0 (no exploration bonus), which empirically let PPO
    converge to a degenerate always-flat policy once it discovered most trades have negative
    expected value net of costs on this data - a well-documented RL-for-trading failure mode,
    not a bug in the env's reward (see app/rl/env.py). A small positive entropy bonus keeps the
    policy from collapsing to zero-activity too early, giving a genuine (if subtle) edge more
    chance to be discovered - it does not manufacture edge that isn't there.

    flat_penalty_bps: forwarded to TradingEnv (see its DEFAULT_FLAT_PENALTY_BPS) - None keeps
    the env's own default.

    reward_mode / dsr_eta: forwarded to TradingEnv (see its module docstring for "pnl" vs
    "differential_sharpe") - None keeps the env's own default for each.

    seed: SB3's own PPO seed, controlling weight initialization and action sampling. Threading a
    distinct seed per call is what makes app/rl/tournament.py's contestants genuinely different
    runs rather than near-duplicates - RL training is seed-sensitive enough that a single run's
    outcome (e.g. "collapsed to 0 trades") is not necessarily representative of what that same
    reward/feature setup can achieve with different random luck.
    """
    env_kwargs = {
        k: v
        for k, v in {"flat_penalty_bps": flat_penalty_bps, "reward_mode": reward_mode, "dsr_eta": dsr_eta}.items()
        if v is not None
    }
    env = TradingEnv(features_df, lookback=lookback, **env_kwargs)
    model = PPO("MlpPolicy", env, verbose=0, ent_coef=ent_coef, seed=seed)

    all_obs, all_actions = [], []
    for teacher in teachers:
        obs, actions = build_labeled_dataset(env, teacher)
        log.info("teacher_dataset_built", teacher=teacher.name, examples=len(obs))
        all_obs.append(obs)
        all_actions.append(actions)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    behavior_clone(model.policy, observations, actions, epochs=epochs)
    return model
