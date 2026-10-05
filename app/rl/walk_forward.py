"""Helpers for walk-forward evaluation of a training RECIPE (scripts/walk_forward.py).

Why walk-forward instead of one holdout: a single validation window is one draw of the market. On
277 models replayed over fresh data, the validation-window score had a NEGATIVE rank correlation
with real out-of-sample return, because the winners were simply the models best fitted to that one
window's regime. Walk-forward trains from scratch at several points spread across years, tests each
on the period right after it, and so measures what actually matters: does "train -> pick the best ->
trade it" make money on average, across regimes, and does the pick beat chance?
"""

import dataclasses

import pandas as pd

from app.strategy.base import Signal, Strategy


@dataclasses.dataclass(frozen=True)
class Fold:
    index: int
    train_end: int  # train = rows [0, train_end)
    select_end: int  # select = rows [train_end, select_end): where candidates are ranked
    test_end: int  # test = rows [select_end, test_end): never seen by training or selection


def fold_bounds(
    n_rows: int, *, folds: int, test_bars: int, select_bars: int, stride: int, min_train_bars: int = 5000
) -> list[Fold]:
    """Fold 0 is the most recent; each older fold's test window sits `stride` bars earlier. Folds
    whose training set would be shorter than min_train_bars are dropped. Returned oldest-first."""
    out = []
    for i in range(folds):
        test_end = n_rows - i * stride
        select_end = test_end - test_bars
        train_end = select_end - select_bars
        if train_end < min_train_bars:
            break
        out.append(Fold(index=i, train_end=train_end, select_end=select_end, test_end=test_end))
    return list(reversed(out))


class VoteStrategy(Strategy):
    """Majority vote of several strategies: trade only when MORE THAN HALF agree on a direction,
    otherwise stay flat. Averaging independently-trained policies cancels a lot of their individual,
    seed-driven noise trades - the question the walk-forward answers is whether what's left is edge."""

    def __init__(self, members: list[Strategy], name: str = "vote"):
        self._members = members
        self.name = name

    def reset(self) -> None:
        for m in self._members:
            if hasattr(m, "reset"):
                m.reset()

    def decide(self, features_row: pd.Series) -> Signal:
        votes = [m.decide(features_row) for m in self._members]  # every member must see every row
        half = len(votes) / 2
        if sum(v == Signal.LONG for v in votes) > half:
            return Signal.LONG
        if sum(v == Signal.SHORT for v in votes) > half:
            return Signal.SHORT
        return Signal.FLAT


def within_fold_rank_correlation(df: pd.DataFrame, score_col: str, outcome_col: str, fold_col: str = "fold") -> float:
    """Rank correlation between a selection-time score and the later out-of-sample outcome, after
    removing each fold's mean from both - so it measures 'does a better score pick a better model
    WITHIN the same market period', not 'were some periods just better than others'."""
    d = df.copy()
    d["_s"] = d[score_col] - d.groupby(fold_col)[score_col].transform("mean")
    d["_o"] = d[outcome_col] - d.groupby(fold_col)[outcome_col].transform("mean")
    return float(d["_s"].rank().corr(d["_o"].rank()))
