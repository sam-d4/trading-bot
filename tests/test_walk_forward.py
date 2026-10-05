import pandas as pd

from app.rl.walk_forward import VoteStrategy, fold_bounds, within_fold_rank_correlation
from app.strategy.base import Signal, Strategy


class _Fixed(Strategy):
    def __init__(self, signal):
        self.name = "fixed"
        self._signal = signal

    def decide(self, features_row):
        return self._signal


def test_fold_bounds_are_disjoint_ordered_and_end_at_the_data_end():
    folds = fold_bounds(31_000, folds=4, test_bars=750, select_bars=750, stride=3000)
    assert len(folds) == 4
    assert folds[-1].test_end == 31_000  # most recent fold is last (oldest first)
    assert [f.index for f in folds] == [3, 2, 1, 0]
    for f in folds:
        assert f.train_end == f.select_end - 750 and f.select_end == f.test_end - 750
    # each older fold's test window sits one stride earlier
    assert folds[-1].test_end - folds[-2].test_end == 3000


def test_fold_bounds_drops_folds_with_too_little_training_data():
    folds = fold_bounds(8_000, folds=6, test_bars=750, select_bars=750, stride=2000, min_train_bars=5000)
    assert all(f.train_end >= 5000 for f in folds) and len(folds) < 6


def test_vote_needs_a_strict_majority():
    row = pd.Series({"x": 0})
    assert VoteStrategy([_Fixed(Signal.LONG), _Fixed(Signal.LONG), _Fixed(Signal.SHORT)]).decide(row) == Signal.LONG
    assert VoteStrategy([_Fixed(Signal.LONG), _Fixed(Signal.SHORT)]).decide(row) == Signal.FLAT  # tie
    assert VoteStrategy([_Fixed(Signal.LONG), _Fixed(Signal.SHORT), _Fixed(Signal.FLAT)]).decide(row) == Signal.FLAT
    assert VoteStrategy([_Fixed(Signal.SHORT)] * 3).decide(row) == Signal.SHORT


def test_rank_correlation_ignores_differences_between_folds():
    # fold B is simply a better period than fold A, but within each fold the score is useless
    df = pd.DataFrame({
        "fold": ["A", "A", "A", "B", "B", "B"],
        "score": [1, 2, 3, 1, 2, 3],
        "outcome": [10, 30, 20, 100, 120, 110],
    })
    useful = pd.DataFrame({"fold": list("AAABBB"), "score": [1, 2, 3, 1, 2, 3], "outcome": [1, 2, 3, 100, 101, 102]})
    assert within_fold_rank_correlation(useful, "score", "outcome") > 0.99
    assert abs(within_fold_rank_correlation(df, "score", "outcome")) < 0.8
