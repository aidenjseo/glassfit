"""Seed catalog tests: frames.json validity, idempotent seeding, and list_frames filters."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from glassfit.catalog.store import ensure_seeded, load_seed
from glassfit.schemas import CatalogFrame
from glassfit.storage.db import connect
from glassfit.storage.repo import Repo

SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "frames.json"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "glassfit.db")
    yield c
    c.close()


@pytest.fixture
def repo(conn: sqlite3.Connection) -> Repo:
    r = Repo(conn)
    ensure_seeded(r, SEED_PATH)
    return r


# --- seed file -------------------------------------------------------------------------------


def test_seed_loads_18_valid_frames() -> None:
    frames = load_seed(SEED_PATH)
    assert len(frames) == 18
    assert all(isinstance(f, CatalogFrame) for f in frames)
    assert len({f.frame_id for f in frames}) == 18  # unique ids
    assert len({f.name for f in frames}) == 18  # unique names


def test_seed_values_within_realistic_ranges() -> None:
    for f in load_seed(SEED_PATH):
        assert 44 <= f.a_mm <= 58, f.frame_id
        assert 32 <= f.b_mm <= 45, f.frame_id
        assert 14 <= f.dbl_mm <= 23, f.frame_id
        assert 135 <= f.temple_mm <= 148, f.frame_id
        assert f.ed_mm >= f.a_mm, f.frame_id
        assert 5 <= f.weight_g <= 30, f.frame_id
        assert f.rim in ("full", "semi", "rimless"), f.frame_id
        assert f.bridge_style in ("keyhole", "saddle", "pad-arm", "double"), f.frame_id
        assert 2 <= len(f.tags) <= 4, f.frame_id
        if f.default_wrap_deg is not None:
            assert 0 <= f.default_wrap_deg <= 15, f.frame_id


def test_seed_has_expected_special_frames() -> None:
    by_id = {f.frame_id: f for f in load_seed(SEED_PATH)}
    assert by_id["GF-012"].low_bridge_fit is True
    assert by_id["GF-003"].spring_hinge is True
    assert by_id["GF-018"].spring_hinge is True
    assert by_id["GF-004"].default_wrap_deg == 7
    assert by_id["GF-007"].default_wrap_deg == 9
    assert by_id["GF-017"].default_wrap_deg == 10


# --- seeding ---------------------------------------------------------------------------------


def test_ensure_seeded_idempotent(repo: Repo, conn: sqlite3.Connection) -> None:
    ensure_seeded(repo, SEED_PATH)  # second run must not duplicate or fail
    count = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert count == 18


def test_ensure_seeded_restores_modified_rows(repo: Repo, conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("UPDATE frames SET name = 'Tampered' WHERE frame_id = 'GF-001'")
    ensure_seeded(repo, SEED_PATH)
    frame = repo.get_frame("GF-001")
    assert frame is not None and frame.name == "Harbor"


# --- queries ---------------------------------------------------------------------------------


def test_get_frame_round_trip(repo: Repo) -> None:
    frame = repo.get_frame("GF-012")
    assert frame is not None
    assert frame.name == "Kai"
    assert frame.low_bridge_fit is True
    assert frame.spring_hinge is False
    assert frame.default_wrap_deg is None
    assert isinstance(frame.tags, list) and "low-bridge-fit" in frame.tags
    assert repo.get_frame("GF-999") is None


def test_list_frames_no_filters_returns_all(repo: Repo) -> None:
    frames = repo.list_frames()
    assert len(frames) == 18
    assert frames == sorted(frames, key=lambda f: f.frame_id)


def test_list_frames_a_min(repo: Repo) -> None:
    ids = {f.frame_id for f in repo.list_frames(a_min=55)}
    assert ids == {"GF-004", "GF-007", "GF-010", "GF-017"}


def test_list_frames_a_max(repo: Repo) -> None:
    ids = {f.frame_id for f in repo.list_frames(a_max=47)}
    assert ids == {"GF-002", "GF-008", "GF-011"}


def test_list_frames_dbl_min(repo: Repo) -> None:
    ids = {f.frame_id for f in repo.list_frames(dbl_min=21)}
    assert ids == {"GF-002", "GF-006", "GF-008", "GF-013"}


def test_list_frames_temple_exact(repo: Repo) -> None:
    frames = repo.list_frames(temple=145)
    assert {f.frame_id for f in frames} == {
        "GF-001",
        "GF-002",
        "GF-006",
        "GF-012",
        "GF-013",
        "GF-014",
        "GF-018",
    }
    assert all(f.temple_mm == 145 for f in frames)


def test_list_frames_combined_filters(repo: Repo) -> None:
    ids = {f.frame_id for f in repo.list_frames(a_min=50, a_max=54, dbl_max=17)}
    assert ids == {"GF-005", "GF-012", "GF-015", "GF-016"}


def test_list_frames_limit(repo: Repo) -> None:
    frames = repo.list_frames(limit=5)
    assert len(frames) == 5
    assert [f.frame_id for f in frames] == [
        "GF-001",
        "GF-002",
        "GF-003",
        "GF-004",
        "GF-005",
    ]
