"""Side-view (head-turn) processing: geometric side classification, iris-ring
scaling, hinge-to-ear refinement, and the full multi-burst scan API flow."""

import pytest
from conftest import FakeBackend, jpeg_upload_files, load_landmark_set, truncate_to_468
from fastapi.testclient import TestClient

from glassfit.measure.side_views import hinge_to_ear_from_side_view, summarize_side_views
from glassfit.schemas import LandmarkSet


def _landmarks(name: str) -> tuple[LandmarkSet, dict]:
    ls, payload = load_landmark_set(name)
    return ls, payload["expected"]


def test_left_turn_exposes_subject_right() -> None:
    ls, expected = _landmarks("synthetic_turn_left")
    assert expected["exposed_side"] == "right"
    result = hinge_to_ear_from_side_view(ls)
    assert result is not None
    side, value = result
    assert side == "right"
    # iris-ring scale is exact by construction; estimate matches the designed geometry
    assert value == pytest.approx(expected["hinge_to_ear_mm"], rel=0.02)


def test_right_turn_exposes_subject_left() -> None:
    ls, expected = _landmarks("synthetic_turn_right")
    result = hinge_to_ear_from_side_view(ls)
    assert result is not None
    assert result[0] == "left" == expected["exposed_side"]


def test_no_iris_returns_none() -> None:
    ls, _ = _landmarks("synthetic_turn_left")
    assert hinge_to_ear_from_side_view(truncate_to_468(ls)) is None


def test_summarize_both_sides() -> None:
    left, left_exp = _landmarks("synthetic_turn_left")
    right, right_exp = _landmarks("synthetic_turn_right")
    counts, medians = summarize_side_views([left, right])
    assert counts == {"right": 1, "left": 1}
    assert medians["right"] == pytest.approx(left_exp["hinge_to_ear_mm"], rel=0.02)
    assert medians["left"] == pytest.approx(right_exp["hinge_to_ear_mm"], rel=0.02)


SIDE_SCRIPT = (
    [{"fixture": "synthetic_average", "pose": (0.0, 0.0, 0.0)}] * 6
    + [{"fixture": "synthetic_turn_left", "pose": (-30.0, 0.0, 0.0)}] * 2
    + [{"fixture": "synthetic_turn_right", "pose": (30.0, 0.0, 0.0)}] * 2
)


def _multi_burst_scan(client: TestClient) -> dict:
    resp = client.post(
        "/api/v1/scan",
        files=jpeg_upload_files(6)
        + [("frames_left", (f"l{i}.jpg", b"fake", "image/jpeg")) for i in range(2)]
        + [("frames_right", (f"r{i}.jpg", b"fake", "image/jpeg")) for i in range(2)],
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_scan_with_head_turn_bursts(make_app) -> None:
    with TestClient(make_app(detector=FakeBackend(script=SIDE_SCRIPT))) as client:
        body = _multi_burst_scan(client)
        side = body["quality"]["side_views"]
        assert side["right_frames"] == 2  # left turn exposes subject-right
        assert side["left_frames"] == 2
        assert side["hinge_to_ear_mm"]["right"] == pytest.approx(65.6, abs=2.0)
        assert side["hinge_to_ear_mm"]["left"] == pytest.approx(65.6, abs=2.0)
        assert body["quality"]["frames_received"] == 10
        assert body["quality"]["frames_used"] == 6  # canonical set is frontal-only

        # recommendation from this scan uses the refined hinge-to-ear values
        rec = client.post(
            "/api/v1/recommendations", json={"pd_mm": 63.0, "scan_id": body["scan_id"]}
        )
        assert rec.status_code == 200, rec.text
        m = rec.json()["measurements"]
        assert m["hinge_to_ear_mm"]["right"] == pytest.approx(
            side["hinge_to_ear_mm"]["right"], abs=1e-6
        )
        assert "hinge_to_ear_refined_from_side_views" in m["quality"]["warnings"]


def test_scan_side_burst_unusable_is_nonblocking(make_app) -> None:
    # side frames come back frontal (no real head turn) -> filtered out, scan still OK
    script = [{"fixture": "synthetic_average", "pose": (0.0, 0.0, 0.0)}] * 10
    with TestClient(make_app(detector=FakeBackend(script=script))) as client:
        body = _multi_burst_scan(client)
    side = body["quality"]["side_views"]
    assert side["right_frames"] == 0 and side["left_frames"] == 0
    assert side["hinge_to_ear_mm"] == {"right": None, "left": None}
    assert any("side_view_right_not_captured" in w for w in body["quality"]["warnings"])
    assert any("side_view_left_not_captured" in w for w in body["quality"]["warnings"])


def test_scan_without_side_bursts_has_no_summary(client: TestClient) -> None:
    resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
    assert resp.status_code == 200
    assert resp.json()["quality"]["side_views"] is None
