"""Round-trip and integrity tests for glassfit.storage against a tmp_path SQLite DB."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from glassfit.errors import NotFound
from glassfit.schemas import (
    AdjustmentsMade,
    AsWorn,
    BehindEar,
    BridgeWidths,
    CatalogFrame,
    Comfort,
    FaceMeasurements,
    FeedbackIn,
    FrameDims,
    FrameReport,
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
from glassfit.storage.db import connect
from glassfit.storage.repo import Repo

# --- builders --------------------------------------------------------------------------------


def _landmarks() -> LandmarkSet:
    points = [(i / 478.0, (i % 100) / 100.0, -0.01) for i in range(478)]
    return LandmarkSet(points=points, image_width=1280, image_height=720)


def _quality() -> ScanQuality:
    return ScanQuality(
        frames_received=6,
        frames_used=4,
        frame_reports=[
            FrameReport(index=0, accepted=True, yaw_deg=1.5, pitch_deg=-2.0, roll_deg=0.5),
            FrameReport(index=1, accepted=False, reject_reason="yaw_exceeded", yaw_deg=15.0),
        ],
        landmark_dispersion_px=2.1,
        ok=True,
        warnings=["slight_roll"],
    )


def _measurements() -> FaceMeasurements:
    return FaceMeasurements(
        mm_per_unit=310.0,
        pd_binocular_mm=63.0,
        pd_monocular_mm=PerSide(right=31.4, left=31.6),
        bridge=BridgeWidths(at_crest_mm=16.0, below_10mm_mm=18.5, below_15mm_mm=20.0),
        bridge_crest_height_mm=8.0,
        zygoma_width_mm=132.0,
        temple_width_mm=138.0,
        face_wrap_radius_mm=95.0,
        cheek_clearance_mm=PerSide(right=6.0, left=5.5),
        hinge_to_ear_mm=PerSide(right=98.0, left=99.0),
        ear_height_asymmetry_mm=1.5,
        behind_ear={
            "right": BehindEar(drop_mm=38.0, angle_deg=42.0),
            "left": BehindEar(drop_mm=39.0, angle_deg=41.0),
        },
        vertex_estimate_mm=12.5,
        canthal_tilt_deg=PerSide(right=4.0, left=3.5),
        pupil_height_ratio=PerSide(right=0.55, left=0.54),
        quality=MeasurementQuality(low_confidence_fields=["hinge_to_ear_mm"]),
    )


def _recommendation(rec_id: str = "rec-1") -> Recommendation:
    return Recommendation(
        recommendation_id=rec_id,
        ruleset_version="rules-1.0.0",
        frame=FrameDims(a_mm=52.0, b_mm=38.0, dbl_mm=18.0, ed_mm=55.0, temple_length_mm=145.0),
        as_worn=AsWorn(pantoscopic_deg=6.5, face_form_deg=6.0, vertex_mm=13.0),
        optics=Optics(
            pd_monocular_mm=PerSide(right=31.4, left=31.6),
            oc_height_mm=PerSide(right=22.0, left=22.5),
            inset_mm=PerSide(right=2.0, left=2.0),
        ),
        nose_pads=NosePads(size="M", splay_deg=30.0, flare_deg=10.0, drop_mm=3.0),
        temples=Temples(
            bend_point_mm_from_hinge=PerSide(right=98.0, left=99.0),
            tip_angle_deg=45.0,
            raise_mm=PerSide(right=0.0, left=1.0),
        ),
        comfort=Comfort(predicted_slip=0.2, nose_pressure=0.3, temple_pressure=0.25),
        notes=["Pantoscopic tilt kept within 5-8 deg."],
        rule_trace={
            "as_worn.pantoscopic_deg": RuleTrace(
                rule_id="panto_default",
                inputs={"canthal_tilt": 4.0},
                raw_value=6.5,
                clamped=False,
            )
        },
    )


def _catalog_frame() -> CatalogFrame:
    return CatalogFrame(
        frame_id="TEST-01",
        name="Test Frame",
        shape="rectangle",
        material="acetate",
        rim="full",
        a_mm=52.0,
        b_mm=38.0,
        dbl_mm=18.0,
        ed_mm=55.0,
        temple_mm=145.0,
        weight_g=22.0,
        bridge_style="keyhole",
        nose_pads="fixed_acetate",
        tags=["test"],
    )


def _save_scan(repo: Repo, scan_id: str = "scan-1") -> str:
    repo.save_scan(
        scan_id,
        _landmarks(),
        _quality(),
        landmark_model="face_landmarker_v2_478",
        app_version="0.1.0",
    )
    return scan_id


def _save_recommendation(repo: Repo, scan_id: str | None, rec_id: str = "rec-1") -> str:
    repo.save_recommendation(
        _recommendation(rec_id),
        scan_id=scan_id,
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=310.0,
        measurements=_measurements(),
        request_extras={"rx": None, "lens_intent": "progressive", "pd_monocular": None},
    )
    return rec_id


# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "glassfit.db")
    yield c
    c.close()


@pytest.fixture
def repo(conn: sqlite3.Connection) -> Repo:
    return Repo(conn)


# --- schema / connection ---------------------------------------------------------------------


def test_connect_creates_schema_and_sets_user_version(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "glassfit.db"
    c = connect(db_path)
    try:
        assert db_path.exists()
        assert c.execute("PRAGMA user_version").fetchone()[0] == 1
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"scans", "recommendations", "frames", "feedback"} <= tables
    finally:
        c.close()


def test_connect_is_idempotent_on_existing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "glassfit.db"
    c1 = connect(db_path)
    Repo(c1).save_scan("scan-x", _landmarks(), _quality(), landmark_model="m", app_version="0.1.0")
    c1.close()
    c2 = connect(db_path)  # must not re-run DDL destructively
    try:
        assert Repo(c2).get_scan("scan-x") is not None
    finally:
        c2.close()


# --- scans -----------------------------------------------------------------------------------


def test_scan_round_trip(repo: Repo) -> None:
    _save_scan(repo)
    got = repo.get_scan("scan-1")
    assert got is not None
    assert got["id"] == "scan-1"
    assert got["app_version"] == "0.1.0"
    assert got["landmark_model"] == "face_landmarker_v2_478"
    assert got["landmarks"] == _landmarks()
    assert got["quality"] == _quality()
    assert "T" in got["created_at"] and "+00:00" in got["created_at"]  # ISO-8601 UTC


def test_get_scan_missing_returns_none(repo: Repo) -> None:
    assert repo.get_scan("nope") is None


# --- recommendations -------------------------------------------------------------------------


def test_recommendation_round_trip(repo: Repo) -> None:
    scan_id = _save_scan(repo)
    _save_recommendation(repo, scan_id)
    got = repo.get_recommendation("rec-1")
    assert got is not None
    assert got["id"] == "rec-1"
    assert got["scan_id"] == scan_id
    assert got["user_id"] == "local"
    assert got["pd_mm"] == 63.0
    assert got["mm_per_unit"] == 310.0
    assert got["measurements"] == _measurements()
    assert got["recommendation"] == _recommendation()
    assert got["request"]["lens_intent"] == "progressive"
    assert got["engine_version"] == "rules-1.0.0"
    assert got["ml_model_version"] is None


def test_recommendation_without_scan_allowed(repo: Repo) -> None:
    _save_recommendation(repo, scan_id=None, rec_id="rec-standalone")
    got = repo.get_recommendation("rec-standalone")
    assert got is not None
    assert got["scan_id"] is None


def test_get_recommendation_missing_returns_none(repo: Repo) -> None:
    assert repo.get_recommendation("nope") is None


# --- feedback --------------------------------------------------------------------------------


def test_feedback_round_trip_and_three_way_join(repo: Repo, conn: sqlite3.Connection) -> None:
    scan_id = _save_scan(repo)
    rec_id = _save_recommendation(repo, scan_id)
    repo.upsert_frames([_catalog_frame()])
    fb_id = repo.save_feedback(
        FeedbackIn(
            recommendation_id=rec_id,
            frame_id="TEST-01",
            nose_pressure=4,
            temple_pressure=2,
            slips=True,
            cheek_touch=False,
            adjustments=AdjustmentsMade(panto_delta_deg=-1.5, vertex_delta_mm=1.0),
            comments="slides down when I look at my phone",
        )
    )
    assert isinstance(fb_id, str) and len(fb_id) == 36  # uuid4

    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (fb_id,)).fetchone()
    assert row["slips"] == 1
    assert row["cheek_touch"] == 0
    assert row["nose_pressure"] == 4
    assert row["user_id"] == "local"
    assert json.loads(row["adjustments_json"])["panto_delta_deg"] == -1.5

    joined = conn.execute(
        "SELECT s.id AS sid, r.id AS rid, f.id AS fid"
        " FROM scans s"
        " JOIN recommendations r ON r.scan_id = s.id"
        " JOIN feedback f ON f.recommendation_id = r.id"
    ).fetchall()
    assert len(joined) == 1
    assert (joined[0]["sid"], joined[0]["rid"], joined[0]["fid"]) == (scan_id, rec_id, fb_id)


def test_feedback_with_frame_worn_dims(repo: Repo, conn: sqlite3.Connection) -> None:
    rec_id = _save_recommendation(repo, scan_id=None, rec_id="rec-fw")
    fb_id = repo.save_feedback(
        FeedbackIn(
            recommendation_id=rec_id,
            frame_worn=FrameDims(
                a_mm=51.0, b_mm=37.0, dbl_mm=19.0, ed_mm=54.0, temple_length_mm=140.0
            ),
            nose_pressure=3,
            temple_pressure=3,
            slips=False,
            cheek_touch=True,
        )
    )
    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (fb_id,)).fetchone()
    assert row["frame_id"] is None
    assert json.loads(row["frame_worn_json"])["a_mm"] == 51.0
    assert row["adjustments_json"] is None
    assert row["comments"] is None


def test_feedback_missing_recommendation_raises_not_found(repo: Repo) -> None:
    with pytest.raises(NotFound):
        repo.save_feedback(
            FeedbackIn(
                recommendation_id="ghost",
                nose_pressure=3,
                temple_pressure=3,
                slips=False,
                cheek_touch=False,
            )
        )


def test_feedback_unknown_frame_raises_not_found(repo: Repo) -> None:
    rec_id = _save_recommendation(repo, scan_id=None, rec_id="rec-uf")
    with pytest.raises(NotFound):
        repo.save_feedback(
            FeedbackIn(
                recommendation_id=rec_id,
                frame_id="NOT-A-FRAME",
                nose_pressure=3,
                temple_pressure=3,
                slips=False,
                cheek_touch=False,
            )
        )


# --- FK / CHECK enforcement (raw SQL, bypassing Repo validation) -----------------------------


def _raw_feedback_insert(conn: sqlite3.Connection, **overrides: object) -> None:
    values: dict[str, object] = {
        "id": "fb-raw",
        "recommendation_id": "rec-1",
        "created_at": "2026-07-27T00:00:00+00:00",
        "nose_pressure": 3,
        "temple_pressure": 3,
        "slips": 0,
        "cheek_touch": 0,
    }
    values.update(overrides)
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with conn:
        conn.execute(f"INSERT INTO feedback ({cols}) VALUES ({marks})", tuple(values.values()))


def test_fk_enforced_on_feedback_recommendation(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _raw_feedback_insert(conn, recommendation_id="does-not-exist")


def test_fk_enforced_on_recommendation_scan(repo: Repo, conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError), conn:
        conn.execute(
            "INSERT INTO recommendations (id, scan_id, created_at, pd_mm, mm_per_unit,"
            " measurements_json, request_json, output_json, engine_version)"
            " VALUES ('r-bad', 'ghost-scan', '2026-07-27T00:00:00+00:00', 63.0, 310.0,"
            " '{}', '{}', '{}', 'rules-1.0.0')"
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"nose_pressure": 0},
        {"nose_pressure": 6},
        {"temple_pressure": 9},
        {"slips": 2},
        {"cheek_touch": -1},
    ],
)
def test_check_constraints_enforced(
    repo: Repo, conn: sqlite3.Connection, overrides: dict[str, object]
) -> None:
    _save_recommendation(repo, scan_id=None, rec_id="rec-1")
    with pytest.raises(sqlite3.IntegrityError):
        _raw_feedback_insert(conn, **overrides)
