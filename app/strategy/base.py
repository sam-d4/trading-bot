from abc import ABC, abstractmethod
from enum import IntEnum

import pandas as pd


class Signal(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


class Strategy(ABC):
    """Common interface for anything that can produce a directional trading signal - a
    rule-based baseline (strategy/baselines.py) or the RL policy (strategy/rl_policy.py).

    Strategies decide DIRECTION ONLY. Position sizing is always the risk layer's job
    (see app/risk/position_sizing.py) - keeping that split means a strategy can never
    accidentally bet more than the risk manager allows.
    """

    name: str

    @abstractmethod
    def decide(self, features_row: pd.Series) -> Signal:
        """Given the most recent feature row (see app/data/features.py), return a Signal."""
        raise NotImplementedError
