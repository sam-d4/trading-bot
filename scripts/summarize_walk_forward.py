"""Re-print the walk-forward report from a saved results CSV (logs/walk_forward_<INSTRUMENT>.csv).

Usage:
    python scripts/summarize_walk_forward.py logs/walk_forward_AUD_USD.csv [more.csv ...]
"""

import sys

import pandas as pd

from app.rl.walk_forward import summarize_folds

for path in sys.argv[1:]:
    print(f"=== {path}")
    print(summarize_folds(pd.read_csv(path)))
    print()
