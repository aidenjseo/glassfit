"""Dataset export: build a real tmp DB via the storage layer, assert the joined frame.

Requires pandas — installed via `uv sync --extra train`; skips cleanly otherwise.
"""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from schema_builders import sample_measurements, sample_recommendation  # noqa: E402

from glassfit.schemas import (  # noqa: E402
    AdjustmentsMade,
    FeedbackIn,
    LandmarkSet,
    ScanQuality,
)
from glassfit.storage.db import connect  # noqa: E402
from glassfit.storage.repo import Repo  # noqa: E402
from glassfit_training.dataset import export, load_training_frame  # noqa: E402


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "glassfit.db"
    conn = connect(path)
    repo = Repo(conn)
    landmarks = LandmarkSet(points=[(0.5, 0.5, 0.0)] * 478, image_width=1280, image_height=720)
    quality = ScanQuality(
        frames_received=6, frames_used=6, frame_reports=[], landmark_dispersion_px=0.5, ok=True
    )
    repo.save_scan("scan1", landmarks, quality, "face_landmarker_v2_478", "0.1.0")
    repo.save_recommendation(
        sample_recommendation("rec1"),
        scan_id="scan1",
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=0.25,
        measurements=sample_measurements(),
        request_extras={"lens_intent": "single_vision_distance"},
    )
    repo.save_feedback(
        FeedbackIn(
            recommendation_id="rec1",
            nose_pressure=2,
            temple_pressure=3,
            slips=False,
            cheek_touch=False,
            adjustments=AdjustmentsMade(panto_delta_deg=1.0),
            comments="ok",
        )
    )
    conn.close()
    return path


def test_join_and_flatten(db_path: Path) -> None:
    frame = load_training_frame(db_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    # provenance
    assert row["engine_version"] == "rules-1.0.0"
    assert row["landmark_model"] == "face_landmarker_v2_478"
    assert row["app_version"] == "0.1.0"
    assert bool(row["has_feedback"]) is True
    # measurement columns (m_*), incl. nested PerSide flattening
    assert row["m_zygoma_width_mm"] == 132.0
    assert row["m_pd_monocular_mm_right"] == 31.4
    assert row["m_behind_ear_right_drop_mm"] == 38.0
    # recommendation columns (rec_*) with rule_trace/notes excluded
    assert row["rec_frame_a_mm"] == 52.0
    assert row["rec_as_worn_pantoscopic_deg"] == 6.5
    assert not any(c.startswith("rec_rule_trace") for c in frame.columns)
    # residual targets + labels
    assert row["adj_panto_delta_deg"] == 1.0
    assert row["nose_pressure"] == 2
    assert bool(row["slips"]) is False


def test_export_csv(db_path: Path, tmp_path: Path) -> None:
    out = export(db_path, tmp_path / "exports" / "train.csv", fmt="csv")
    assert out.exists()
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert "m_zygoma_width_mm" in frame.columns


def test_recommendation_without_feedback_kept(db_path: Path) -> None:
    conn = connect(db_path)
    repo = Repo(conn)
    rec = sample_recommendation("rec2")
    repo.save_recommendation(
        rec,
        scan_id="scan1",
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=0.25,
        measurements=sample_measurements(),
        request_extras={},
    )
    conn.close()
    frame = load_training_frame(db_path)
    assert len(frame) == 2
    assert set(frame["has_feedback"]) == {True, False}
