"""Dataset export: build a real tmp DB via the storage layer, assert the joined frame.

Requires pandas — installed via `uv sync --extra train`; skips cleanly otherwise.
"""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from glassfit.schemas import (  # noqa: E402
    AdjustmentsMade,
    AsWorn,
    BehindEar,
    BridgeWidths,
    Comfort,
    FaceMeasurements,
    FeedbackIn,
    FrameDims,
    LandmarkSet,
    MeasurementQuality,
    NosePads,
    Optics,
    PerSide,
    Recommendation,
    RuleTrace,
    ScanQuality,
    Temples,
)
from glassfit.storage.db import connect  # noqa: E402
from glassfit.storage.repo import Repo  # noqa: E402
from glassfit_training.dataset import export, load_training_frame  # noqa: E402


def _side(v: float) -> PerSide:
    return PerSide(right=v, left=v)


def _measurements() -> FaceMeasurements:
    return FaceMeasurements(
        mm_per_unit=0.25,
        pd_binocular_mm=63.0,
        pd_monocular_mm=_side(31.5),
        bridge=BridgeWidths(at_crest_mm=18.0, below_10mm_mm=17.5, below_15mm_mm=19.0),
        bridge_crest_height_mm=14.0,
        zygoma_width_mm=130.0,
        temple_width_mm=118.0,
        face_wrap_radius_mm=70.0,
        cheek_clearance_mm=_side(21.5),
        hinge_to_ear_mm=_side(54.5),
        ear_height_asymmetry_mm=0.0,
        behind_ear={
            "right": BehindEar(drop_mm=14.0, angle_deg=29.0),
            "left": BehindEar(drop_mm=14.0, angle_deg=29.0),
        },
        vertex_estimate_mm=12.0,
        canthal_tilt_deg=_side(1.1),
        pupil_height_ratio=_side(0.54),
        quality=MeasurementQuality(low_confidence_fields=["hinge_to_ear_mm"]),
    )


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec1",
        ruleset_version="1.0.0",
        frame=FrameDims(a_mm=52, b_mm=38, dbl_mm=18, ed_mm=55, temple_length_mm=145),
        as_worn=AsWorn(pantoscopic_deg=6.5, face_form_deg=7.0, vertex_mm=13.0),
        optics=Optics(pd_monocular_mm=_side(31.5), oc_height_mm=_side(19.0), inset_mm=_side(1.2)),
        nose_pads=NosePads(size="M", splay_deg=25, flare_deg=10, drop_mm=2),
        temples=Temples(
            bend_point_mm_from_hinge=_side(85.0), tip_angle_deg=35.0, raise_mm=_side(0.0)
        ),
        comfort=Comfort(predicted_slip=0.2, nose_pressure=0.3, temple_pressure=0.25),
        notes=["Raise left temple 1.5 mm"],
        rule_trace={
            "frame.a_mm": RuleTrace(
                rule_id="a_from_zygoma",
                inputs={"zygoma_width_mm": 130.0},
                raw_value=52.0,
                clamped=False,
            )
        },
    )


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
        _recommendation(),
        scan_id="scan1",
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=0.25,
        measurements=_measurements(),
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
    assert row["engine_version"] == "1.0.0"
    assert row["landmark_model"] == "face_landmarker_v2_478"
    assert row["app_version"] == "0.1.0"
    assert bool(row["has_feedback"]) is True
    # measurement columns (m_*), incl. nested PerSide flattening
    assert row["m_zygoma_width_mm"] == 130.0
    assert row["m_pd_monocular_mm_right"] == 31.5
    assert row["m_behind_ear_right_drop_mm"] == 14.0
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
    rec = _recommendation().model_copy(update={"recommendation_id": "rec2"})
    repo.save_recommendation(
        rec,
        scan_id="scan1",
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=0.25,
        measurements=_measurements(),
        request_extras={},
    )
    conn.close()
    frame = load_training_frame(db_path)
    assert len(frame) == 2
    assert set(frame["has_feedback"]) == {True, False}
