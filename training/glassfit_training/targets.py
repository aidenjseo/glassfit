"""Training targets.

- Ratings model: ``rating`` (1-5) as a regression target — ordinal, and MAE in
  rating points is the honest metric.
- Comfort models: ``nose_pressure``/``temple_pressure`` (1-5 regressors) and
  ``slips`` (binary classifier). Adjustment deltas (``adj_*``) become residual
  regression targets once enough non-null rows exist.
"""

from typing import Any

COMFORT_REGRESSION = ("nose_pressure", "temple_pressure")
COMFORT_BINARY = ("slips", "cheek_touch")


def rating_target(frame: Any):
    return frame["rating"].astype(float)


def comfort_targets(labeled: Any) -> dict[str, Any]:
    """Available comfort targets from an already-label-filtered feedback frame."""
    targets: dict[str, Any] = {}
    for name in COMFORT_REGRESSION:
        series = labeled[name].dropna().astype(float)
        if len(series):
            targets[name] = series
    for name in COMFORT_BINARY:
        series = labeled[name].dropna().astype(int)
        if series.nunique() > 1:  # a classifier needs both classes present
            targets[name] = series
    return targets
