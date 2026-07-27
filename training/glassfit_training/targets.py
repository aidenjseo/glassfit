"""Phase 3 stub — training targets for the residual model.

- Regression: ``adj_*`` adjustment deltas (what a fitter actually changed vs the rules
  output) — one HistGradientBoostingRegressor per target.
- Classification: ``slips``, ``cheek_touch`` — HistGradientBoostingClassifier.
- Ordinal-as-regression: ``nose_pressure``, ``temple_pressure`` (1-5).

Serving-time correction: ``final = rules_output + clamp(residual_prediction)`` with
clamps keeping every corrected value inside rule-safe bands (e.g. pantoscopic 3-10°).
"""

from typing import Any


def build_targets(frame: Any) -> Any:
    """DataFrame -> {target_name: y}. Not implemented until Phase 3."""
    raise NotImplementedError("Phase 3")
