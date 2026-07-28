"""Round-trip and integrity tests for glassfit.storage against a tmp_path SQLite DB."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from schema_builders import sample_measurements, sample_recommendation

from glassfit.errors import NotFound
from glassfit.schemas import (
    AdjustmentsMade,
    CatalogFrame,
    FeedbackIn,
    FrameDims,
    FrameReport,
    LandmarkSet,
    MatchRatingIn,
    ScanQuality,
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
        sample_recommendation(rec_id),
        scan_id=scan_id,
        user_id="local",
        pd_mm=63.0,
        mm_per_unit=310.0,
        measurements=sample_measurements(),
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
        assert c.execute("PRAGMA user_version").fetchone()[0] == 3
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"scans", "recommendations", "frames", "feedback", "match_ratings"} <= tables
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
    assert got["measurements"] == sample_measurements()
    assert got["recommendation"] == sample_recommendation()
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


# --- schema v2: match ratings ----------------------------------------------------------------


def test_fresh_db_is_at_current_schema_with_match_ratings(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "match_ratings" in tables


def test_v1_db_upgrades_in_place_preserving_data(tmp_path: Path) -> None:
    path = tmp_path / "upgrade.db"
    conn = connect(path)
    repo = Repo(conn)
    _save_recommendation(repo, scan_id=None, rec_id="rec-old")
    # simulate a database created before migration v2 shipped
    conn.execute("DROP TABLE match_ratings")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    upgraded = connect(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == 3
    assert upgraded.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0] == 1
    Repo(upgraded).save_match_rating(
        MatchRatingIn(recommendation_id="rec-old", frame_id=_seed_frame(upgraded), rating=5)
    )
    assert upgraded.execute("SELECT COUNT(*) FROM match_ratings").fetchone()[0] == 1
    upgraded.close()


def _seed_frame(conn: sqlite3.Connection) -> str:
    Repo(conn).upsert_frames([_catalog_frame()])
    return "TEST-01"


def test_match_rating_round_trip_and_not_found(repo: Repo, conn: sqlite3.Connection) -> None:
    _save_recommendation(repo, scan_id=None, rec_id="rec-1")
    repo.upsert_frames([_catalog_frame()])
    rating_id = repo.save_match_rating(
        MatchRatingIn(
            recommendation_id="rec-1",
            frame_id="TEST-01",
            rating=4,
            fit_score=0.82,
            components={"bridge_fit": 0.9, "shape_affinity": 0.7},
            comment="nice",
        )
    )
    row = conn.execute("SELECT * FROM match_ratings WHERE id = ?", (rating_id,)).fetchone()
    assert row["rating"] == 4
    assert row["fit_score"] == 0.82
    assert json.loads(row["components_json"]) == {"bridge_fit": 0.9, "shape_affinity": 0.7}

    with pytest.raises(NotFound):
        repo.save_match_rating(
            MatchRatingIn(recommendation_id="ghost", frame_id="TEST-01", rating=3)
        )
    with pytest.raises(NotFound):
        repo.save_match_rating(
            MatchRatingIn(recommendation_id="rec-1", frame_id="GHOST-99", rating=3)
        )


def test_match_rating_upserts_latest_wins(repo: Repo, conn: sqlite3.Connection) -> None:
    _save_recommendation(repo, scan_id=None, rec_id="rec-1")
    repo.upsert_frames([_catalog_frame()])
    first = repo.save_match_rating(
        MatchRatingIn(recommendation_id="rec-1", frame_id="TEST-01", rating=2)
    )
    second = repo.save_match_rating(
        MatchRatingIn(recommendation_id="rec-1", frame_id="TEST-01", rating=5, comment="better")
    )
    assert second == first  # the surviving row keeps its identity
    rows = conn.execute("SELECT rating, comment FROM match_ratings").fetchall()
    assert len(rows) == 1
    assert rows[0]["rating"] == 5 and rows[0]["comment"] == "better"


def test_v2_dup_ratings_deduped_by_v3_migration(tmp_path: Path) -> None:
    path = tmp_path / "dups.db"
    conn = connect(path)
    repo = Repo(conn)
    _save_recommendation(repo, scan_id=None, rec_id="rec-1")
    repo.upsert_frames([_catalog_frame()])
    # simulate a v2 database that accumulated duplicate rows before uniqueness shipped
    conn.execute("DROP INDEX uq_match_ratings_pair")
    conn.execute("PRAGMA user_version = 2")
    for i, rating in enumerate([2, 4]):
        conn.execute(
            "INSERT INTO match_ratings (id, recommendation_id, frame_id, created_at,"
            " user_id, rating) VALUES (?, 'rec-1', 'TEST-01', ?, 'local', ?)",
            (f"r-{i}", f"2026-01-0{i + 1}T00:00:00+00:00", rating),
        )
    conn.commit()
    conn.close()

    upgraded = connect(path)
    rows = upgraded.execute("SELECT rating FROM match_ratings").fetchall()
    assert len(rows) == 1
    assert rows[0]["rating"] == 4  # the later row survived
    upgraded.close()
