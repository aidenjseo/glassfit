"""Phase 3 stub — evaluation gate for the residual model.

Cross-validated comparison of rules-only vs rules+residual on held-out feedback rows;
the residual model ships only if it beats the rules baseline. Reports per-target MAE
(deltas), AUC (slips/cheek touch), and calibration of the comfort proxies.
"""

from typing import Any


def evaluate(frame: Any) -> dict:
    raise NotImplementedError("Phase 3")
