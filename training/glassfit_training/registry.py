"""Model artifact registry.

Layout: ``training/artifacts/<target>/<UTC-timestamp>/model.joblib`` + ``metadata.json``
(rows, features, CV metrics, baselines, label provenance, engine versions). The newest
artifact per target wins at load time.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def save_model(target: str, model: Any, metadata: dict) -> Path:
    import joblib

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_DIR / target / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return out_dir


def load_latest(target: str) -> tuple[Any, dict] | None:
    """Newest saved (model, metadata) for ``target``, or None when nothing is saved."""
    import joblib

    target_dir = ARTIFACTS_DIR / target
    if not target_dir.is_dir():
        return None
    runs = sorted(
        (d for d in target_dir.iterdir() if (d / "model.joblib").exists()), key=lambda d: d.name
    )
    if not runs:
        return None
    latest = runs[-1]
    metadata = json.loads((latest / "metadata.json").read_text())
    return joblib.load(latest / "model.joblib"), metadata
