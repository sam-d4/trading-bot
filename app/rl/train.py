"""PPO fine-tuning entrypoint. Always continues from an already-informed policy - either the
behavior-cloning warm start (pretrain.py, first run) or the current live model's own weights
(subsequent retrain cycles) - never a randomly initialized one, per the plan's cold-start
approach.
"""

from pathlib import Path

from stable_baselines3 import PPO

from app.rl.env import TradingEnv


def fine_tune(
    model: PPO,
    features_df,
    *,
    total_timesteps: int,
    lookback: int = 20,
    flat_penalty_bps: float | None = None,
    reward_mode: str | None = None,
    dsr_eta: float | None = None,
) -> PPO:
    env_kwargs = {
        k: v
        for k, v in {"flat_penalty_bps": flat_penalty_bps, "reward_mode": reward_mode, "dsr_eta": dsr_eta}.items()
        if v is not None
    }
    env = TradingEnv(features_df, lookback=lookback, **env_kwargs)
    model.set_env(env)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    return model


def save_model(model: PPO, models_dir: Path, name: str) -> str:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.zip"
    model.save(str(path))
    return str(path)


def load_model(path: str) -> PPO:
    return PPO.load(path)
