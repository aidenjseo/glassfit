"""Phase 3 stub — model artifact registry.

Layout: ``training/artifacts/<target>/<timestamp>/model.joblib`` + ``metadata.json``
(train-set size, engine_version range, CV metrics, feature list). The backend loads
the newest artifact whose engine_version matches the running rules engine and stamps
``ml_model_version`` on recommendations it corrects.
"""

from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def save_model(target: str, model: Any, metadata: dict) -> Path:
    raise NotImplementedError("Phase 3")


def load_latest(target: str, engine_version: str) -> Any:
    raise NotImplementedError("Phase 3")
