"""Wraps a trained SB3 PPO model behind the same Strategy interface the baseline strategies use
(app/strategy/base.py), so the backtest engine, the validation gate, and the live TradingEngine
can all treat an RL policy exactly like a rule-based strategy - decide() takes one feature row
per completed bar.

Internally buffers a rolling window of feature rows matching app/rl/env.py's `lookback`, since
the RL policy's observation is a stacked window, not a single row - callers never see that.
Relies on decide() being called in chronological order (true for both the backtest engine's
row-wise apply and the live engine's per-bar calls) - out-of-order calls would corrupt the
window, unlike the stateless baseline strategies.

feature_columns must be passed explicitly (not derived from a global constant) and must exactly
match what the wrapped model was trained on - see app/data/features.py's feature_columns_for.
Get this wrong (e.g. load a EUR_USD model but pass GBP_USD's column list) and predict() either
raises on a shape mismatch or, worse, silently feeds the model numbers from the wrong columns in
the wrong slots. app/core/engine.py resolves this from the model's own
persistence.models.ModelVersion.instrument record, never guesses it.
"""

import collections
import os

# Must be set before stable_baselines3 (which imports matplotlib) is imported - see
# app/rl/__init__.py for why. Duplicated here (idempotent) since this module can be imported
# directly without going through the app.rl package first.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from app.strategy.base import Signal, Strategy

ACTION_TO_SIGNAL = {0: Signal.FLAT, 1: Signal.LONG, 2: Signal.SHORT}
SIGNAL_TO_POSITION = {Signal.FLAT: 0, Signal.LONG: 1, Signal.SHORT: -1}


class RLPolicyStrategy(Strategy):
    def __init__(self, model: PPO, feature_columns: list[str], *, lookback: int = 20, model_version_name: str = "rl_policy"):
        self.name = f"rl_policy:{model_version_name}"
        self._model = model
        self._feature_columns = feature_columns
        self._lookback = lookback
        self._buffer: collections.deque = collections.deque(maxlen=lookback)
        self._position = 0

    @classmethod
    def load(
        cls, artifact_path: str, feature_columns: list[str], *, lookback: int = 20, model_version_name: str = "rl_policy"
    ) -> "RLPolicyStrategy":
        model = PPO.load(artifact_path)
        return cls(model, feature_columns, lookback=lookback, model_version_name=model_version_name)

    def reset(self) -> None:
        self._buffer.clear()
        self._position = 0

    def decide(self, features_row: pd.Series) -> Signal:
        self._buffer.append(features_row[self._feature_columns].to_numpy(dtype=np.float32))
        if len(self._buffer) < self._lookback:
            return Signal.FLAT  # not enough history yet to form a full observation window

        window = np.concatenate(list(self._buffer))
        obs = np.concatenate([window, [float(self._position)]]).astype(np.float32)
        action, _ = self._model.predict(obs, deterministic=True)
        signal = ACTION_TO_SIGNAL[int(action)]
        self._position = SIGNAL_TO_POSITION[signal]
        return signal
