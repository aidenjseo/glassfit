"""API tests over the full wired app (fake detector, temp DB, no mediapipe)."""

import json

from conftest import FakeBackend, jpeg_upload_files, load_landmark_fixture
from fastapi.testclient import TestClient


def _do_scan(client: TestClient) -> dict:
    resp = client.post(
        "/api/v1/scan",
        files=jpeg_upload_files(6),
        data={"meta": json.dumps({"width": 1280, "height": 720})},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["ruleset_version"] == "1.0.0"
    assert "mediapipe_available" in body


def test_scan_happy_path(client: TestClient) -> None:
    body = _do_scan(client)
    assert len(body["landmarks"]["points"]) == 478
    assert len(body["landmarks_px"]) == 478
    assert body["quality"]["frames_used"] == 6
    assert body["quality"]["ok"] is True
    names = {seg["name"] for seg in body["overlay_segments"]}
    assert {"pd", "zygoma_width", "temple_width", "bridge_crest"} <= names
    assert "pupil_right" in body["key_points"]
    # subject-right pupil must sit at smaller x than subject-left (unmirrored frame)
    assert body["key_points"]["pupil_right"][0] < body["key_points"]["pupil_left"][0]


def test_scan_then_recommendation_roundtrip(client: TestClient) -> None:
    scan = _do_scan(client)
    resp = client.post(
        "/api/v1/recommendations",
        json={"pd_mm": 63.0, "scan_id": scan["scan_id"], "lens_intent": "single_vision_distance"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 5.0 <= body["as_worn"]["pantoscopic_deg"] <= 8.0
    assert 5.0 <= body["as_worn"]["face_form_deg"] <= 10.0
    assert 12.0 <= body["as_worn"]["vertex_mm"] <= 14.0
    assert abs(body["measurements"]["pd_binocular_mm"] - 63.0) < 0.01
    assert abs(body["measurements"]["zygoma_width_mm"] - 130.0) < 0.01
    assert body["rule_trace"]
    assert body["notes"]
    assert body["scan_id"] == scan["scan_id"]

    # feedback joins back to the persisted recommendation
    fb = client.post(
        "/api/v1/feedback",
        json={
            "recommendation_id": body["recommendation_id"],
            "nose_pressure": 2,
            "temple_pressure": 3,
            "slips": False,
            "cheek_touch": False,
            "adjustments": {"panto_delta_deg": 1.0},
            "comments": "slight pinch",
        },
    )
    assert fb.status_code == 201, fb.text
    assert fb.json()["feedback_id"]


def test_recommendation_with_inline_landmarks(client: TestClient) -> None:
    fixture = load_landmark_fixture("synthetic_average")
    resp = client.post(
        "/api/v1/recommendations",
        json={"pd_mm": fixture["pd_mm_ground_truth"], "landmarks": fixture["landmark_set"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["scan_id"] is None


def test_measurements_endpoint(client: TestClient) -> None:
    fixture = load_landmark_fixture("synthetic_average")
    resp = client.post(
        "/api/v1/measurements",
        json={"pd_mm": 63.0, "landmarks": fixture["landmark_set"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert abs(body["zygoma_width_mm"] - 130.0) < 0.01
    assert "hinge_to_ear_mm" in body["quality"]["low_confidence_fields"]


def test_frames_seeded_and_filterable(client: TestClient) -> None:
    assert len(client.get("/api/v1/frames").json()["frames"]) == 18
    narrow = client.get("/api/v1/frames", params={"dbl_min": 20}).json()["frames"]
    assert 0 < len(narrow) < 18
    assert all(f["dbl_mm"] >= 20 for f in narrow)


def test_error_envelopes(client: TestClient) -> None:
    resp = client.post("/api/v1/recommendations", json={"pd_mm": 63.0, "scan_id": "nope"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"

    resp = client.post("/api/v1/recommendations", json={"pd_mm": 30.0, "scan_id": "x"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    # exactly-one-of source rule
    resp = client.post("/api/v1/recommendations", json={"pd_mm": 63.0})
    assert resp.status_code == 422


def test_scan_no_face(make_app) -> None:
    with TestClient(make_app(detector=FakeBackend(face_count=0))) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "NO_FACE_DETECTED"


def test_scan_multiple_faces(make_app) -> None:
    with TestClient(make_app(detector=FakeBackend(face_count=2))) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "MULTIPLE_FACES"


def test_scan_poor_quality_on_yaw(make_app) -> None:
    with TestClient(make_app(detector=FakeBackend(pose=(30.0, 0.0, 0.0)))) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(6))
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "POOR_SCAN_QUALITY"
    assert body["details"]["frame_reports"][0]["reject_reason"] == "yaw_exceeded"


def test_scan_detector_unavailable(make_app, tmp_path) -> None:
    # no detector override + model path pointing nowhere -> 503 regardless of
    # whether mediapipe itself is installed in this environment
    app = make_app(landmarker_model_path=tmp_path / "missing.task")
    with TestClient(app) as client:
        resp = client.post("/api/v1/scan", files=jpeg_upload_files(3))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "MEDIAPIPE_UNAVAILABLE"
