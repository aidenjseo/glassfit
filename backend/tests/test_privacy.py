"""Privacy guarantee: raw camera frames never reach disk unless explicitly opted in."""

from pathlib import Path

from conftest import FakeBackend, jpeg_upload_files
from fastapi.testclient import TestClient


def _image_files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in {".jpg", ".jpeg", ".png"}]


def test_scan_persists_no_frame_bytes_by_default(make_app, tmp_path: Path) -> None:
    with TestClient(make_app(detector=FakeBackend())) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
        assert resp.status_code == 200
    assert _image_files_under(tmp_path) == []
    # only the SQLite database (and its WAL/journal) may exist
    suffixes = {p.suffix for p in tmp_path.rglob("*") if p.is_file()}
    assert suffixes <= {".db", ".db-wal", ".db-shm", ".db-journal", ""}


def test_save_frames_debug_opt_in(make_app, tmp_path: Path) -> None:
    with TestClient(make_app(detector=FakeBackend(), save_frames=True)) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
        assert resp.status_code == 200
    saved = _image_files_under(tmp_path / "data" / "frames")
    assert len(saved) == 6
