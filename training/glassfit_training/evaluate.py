"""Cross-validated evaluation with honest baselines.

A model only counts as "working" when it beats the baselines that need no
learning at all:
- ratings:  (a) predict the global mean rating, (b) the matcher's own heuristic
            ``1 + 4 * fit_score`` — the model must out-predict the very score
            it was seeded from, i.e. it learned something beyond it.
- comfort:  the global mean (regression) / majority class (classification).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# The load-bearing convention linking a 0..1 score to the 1-5 rating scale —
# shared by the baseline below, the synthetic labeler, and the training tests.
RATING_MIN = 1.0
RATING_SPAN = 4.0


def heuristic_rating(score01):
    """Map a 0..1 fit-like score onto the 1-5 rating scale (scalar or vector)."""
    return RATING_MIN + RATING_SPAN * score01


def _default_n_splits(y) -> int:
    return max(3, min(5, len(y) // 20))


def crossval_regression(model_factory, x, y, n_splits: int | None = None) -> dict[str, float]:
    n_splits = n_splits or _default_n_splits(y)
    errors, baseline_errors = [], []
    for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=7).split(x):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        errors.append(np.abs(pred - y.iloc[test_idx]).mean())
        baseline_errors.append(np.abs(y.iloc[train_idx].mean() - y.iloc[test_idx]).mean())
    return {
        "cv_mae": float(np.mean(errors)),
        "baseline_mean_mae": float(np.mean(baseline_errors)),
        "n_splits": n_splits,
    }


def rating_heuristic_mae(frame: Any, y) -> float | None:
    """MAE of the no-learning heuristic (None when fit_score is entirely missing)."""
    fit = pd.to_numeric(frame.get("fit_score"), errors="coerce")
    mask = fit.notna()
    if not mask.any():
        return None
    return float(np.abs(heuristic_rating(fit[mask]) - y[mask]).mean())


def crossval_classification(model_factory, x, y, n_splits: int | None = None) -> dict[str, float]:
    n_splits = n_splits or _default_n_splits(y)
    accs, base = [], []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    for train_idx, test_idx in folds.split(x, y):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        accs.append(float((model.predict(x.iloc[test_idx]) == y.iloc[test_idx]).mean()))
        majority = y.iloc[train_idx].mode()[0]
        base.append(float((y.iloc[test_idx] == majority).mean()))
    return {
        "cv_accuracy": float(np.mean(accs)),
        "baseline_majority_accuracy": float(np.mean(base)),
        "n_splits": n_splits,
    }
