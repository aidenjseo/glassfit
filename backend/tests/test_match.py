"""Frame matching: hard filters, weighted scoring, flags, and the API flow."""

import json

from conftest import FakeBackend, jpeg_upload_files
from fastapi.testclient import TestClient

from glassfit.catalog.match import match_frames, score_frame
from glassfit.schemas import CatalogFrame, FrameDims


def frame(**overrides) -> CatalogFrame:
    base = {
        "frame_id": "F-1",
        "name": "Test",
        "shape": "rectangle",
        "material": "acetate",
        "rim": "full",
        "a_mm": 52.0,
        "b_mm": 38.0,
        "dbl_mm": 18.0,
        "ed_mm": 55.0,
        "temple_mm": 145.0,
        "weight_g": 22.0,
        "bridge_style": "keyhole",
        "nose_pads": "fixed_acetate",
        "tags": [],
    }
    return CatalogFrame(**{**base, **overrides})


TARGETS = FrameDims(a_mm=52.0, b_mm=38.0, dbl_mm=18.0, ed_mm=55.0, temple_length_mm=145.0)


def test_perfect_match_scores_one_with_zero_deltas() -> None:
    match = score_frame(frame(), TARGETS)
    assert match is not None
    assert match.fit_score == 1.0
    assert all(v == 0.0 for v in match.deltas.values())


def test_dbl_hard_filter() -> None:
    assert score_frame(frame(dbl_mm=20.5), TARGETS) is None  # +2.5 > ±2 tolerance
    assert score_frame(frame(dbl_mm=15.5), TARGETS) is None
    assert score_frame(frame(dbl_mm=20.0), TARGETS) is not None  # boundary passes


def test_temple_too_short_filtered_but_long_ok() -> None:
    assert score_frame(frame(temple_mm=140.0), TARGETS) is None  # 5mm short
    long = score_frame(frame(temple_mm=150.0), TARGETS)
    assert long is not None  # longer than needed is fine (bend further back)


def test_deltas_are_frame_minus_target() -> None:
    match = score_frame(frame(a_mm=54.0, dbl_mm=17.0), TARGETS)
    assert match is not None
    assert match.deltas["a_mm"] == 2.0
    assert match.deltas["dbl_mm"] == -1.0


def test_ranking_prefers_closer_dimensions() -> None:
    catalog = [
        frame(frame_id="close", a_mm=53.0),
        frame(frame_id="exact"),
        frame(frame_id="far", a_mm=56.0, dbl_mm=19.5),
        frame(frame_id="rejected", dbl_mm=24.0),
    ]
    matches = match_frames(catalog, TARGETS)
    assert [m.frame.frame_id for m in matches] == ["exact", "close", "far"]
    assert matches[0].fit_score > matches[1].fit_score > matches[2].fit_score


def test_low_bridge_preference_bonus_and_flags() -> None:
    # imperfect frames so the penalty reduction is visible in the score
    fixed = score_frame(frame(frame_id="fixed", a_mm=53.0), TARGETS, prefer_low_bridge=True)
    adjustable = score_frame(
        frame(frame_id="adj", a_mm=53.0, nose_pads="adjustable"), TARGETS, prefer_low_bridge=True
    )
    assert fixed is not None and adjustable is not None
    assert adjustable.fit_score > fixed.fit_score
    assert any("Low-bridge friendly" in f for f in adjustable.flags)
    assert any("may slide" in f for f in fixed.flags)


def test_low_bridge_bonus_cannot_saturate_ranking() -> None:
    # Regression: an additive bonus used to clamp both frames to 1.0, letting the
    # WORSE frame win on the alphabetical tie-break.
    catalog = [
        frame(frame_id="A-1", a_mm=53.0, nose_pads="adjustable"),  # 1mm off
        frame(frame_id="Z-9", nose_pads="adjustable"),  # perfect match
    ]
    matches = match_frames(catalog, TARGETS, prefer_low_bridge=True)
    assert [m.frame.frame_id for m in matches] == ["Z-9", "A-1"]
    assert matches[0].fit_score == 1.0
    assert matches[1].fit_score < 1.0  # imperfect frames never display as 100%


def test_limit_and_score_bounds() -> None:
    catalog = [frame(frame_id=f"F-{i}", a_mm=52.0 + i * 0.5) for i in range(10)]
    matches = match_frames(catalog, TARGETS, limit=3)
    assert len(matches) == 3
    assert all(0.0 <= m.fit_score <= 1.0 for m in matches)


# --- API flow -------------------------------------------------------------------------------


def test_match_from_recommendation(client: TestClient) -> None:
    scan = client.post("/api/v1/scan", files=jpeg_upload_files(6)).json()
    rec = client.post(
        "/api/v1/recommendations", json={"pd_mm": 63.0, "scan_id": scan["scan_id"]}
    ).json()
    resp = client.post(
        "/api/v1/frames/match", json={"recommendation_id": rec["recommendation_id"], "limit": 5}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["targets"] == rec["frame"]
    assert 0 < len(body["matches"]) <= 5
    scores = [m["fit_score"] for m in body["matches"]]
    assert scores == sorted(scores, reverse=True)
    for m in body["matches"]:
        assert abs(m["deltas"]["dbl_mm"]) <= 2.0  # hard filter held

    # feedback can reference a matched catalog frame (the ML training join)
    fb = client.post(
        "/api/v1/feedback",
        json={
            "recommendation_id": rec["recommendation_id"],
            "frame_id": body["matches"][0]["frame"]["frame_id"],
            "nose_pressure": 2,
            "temple_pressure": 2,
            "slips": False,
            "cheek_touch": False,
        },
    )
    assert fb.status_code == 201, fb.text


def test_match_with_inline_targets(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/frames/match",
        json={"targets": json.loads(TARGETS.model_dump_json())},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["matches"]) > 0


def test_match_validation_and_unknown_recommendation(client: TestClient) -> None:
    resp = client.post("/api/v1/frames/match", json={})
    assert resp.status_code == 422
    resp = client.post("/api/v1/frames/match", json={"recommendation_id": "nope"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_match_uses_low_bridge_preference_from_measurements(make_app) -> None:
    # The synthetic fixture has a HIGH crest (14mm) -> no low-bridge preference expected;
    # this asserts the wiring reads the stored measurement rather than defaulting on.
    with TestClient(make_app(detector=FakeBackend())) as client:
        scan = client.post("/api/v1/scan", files=jpeg_upload_files(6)).json()
        rec = client.post(
            "/api/v1/recommendations", json={"pd_mm": 63.0, "scan_id": scan["scan_id"]}
        ).json()
        body = client.post(
            "/api/v1/frames/match", json={"recommendation_id": rec["recommendation_id"]}
        ).json()
    for m in body["matches"]:
        assert not any("Low-bridge friendly" in f for f in m["flags"])
