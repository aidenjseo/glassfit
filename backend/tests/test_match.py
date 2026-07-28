"""Component-based frame matching: hard filters, per-component behavior, ratings API."""

import json

from conftest import FakeBackend, jpeg_upload_files
from fastapi.testclient import TestClient
from schema_builders import sample_measurements

from glassfit.catalog.match import NEUTRAL, MatchContext, match_frames, score_frame
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
DIMS_ONLY = MatchContext(targets=TARGETS)


def measured_ctx(**measurement_overrides) -> MatchContext:
    # sample_measurements: PD 63, temple width 138, crest 8, cheeks ~5.75mm clearance,
    # face 142/132 (balanced), jaw ratio 98/132=0.74 (non-angular), canthal +4deg
    return MatchContext(
        targets=TARGETS,
        measurements=sample_measurements(**measurement_overrides),
        face_form_target_deg=6.0,
    )


# --- hard filters -----------------------------------------------------------------------------


def test_dbl_hard_filter_fixed_vs_adjustable_pads() -> None:
    assert score_frame(frame(dbl_mm=20.5), DIMS_ONLY) is None  # +2.5 > ±2 fixed
    assert score_frame(frame(dbl_mm=20.5, nose_pads="adjustable"), DIMS_ONLY) is not None
    assert score_frame(frame(dbl_mm=21.5, nose_pads="adjustable"), DIMS_ONLY) is None  # > ±3


def test_temple_too_short_filtered_but_long_ok() -> None:
    assert score_frame(frame(temple_mm=140.0), DIMS_ONLY) is None  # 5mm short
    assert score_frame(frame(temple_mm=150.0), DIMS_ONLY) is not None


# --- components -------------------------------------------------------------------------------


def test_centration_rewards_compensating_a_and_dbl() -> None:
    compensating = score_frame(frame(a_mm=53.0, dbl_mm=17.0), DIMS_ONLY)  # sum unchanged
    uncompensated = score_frame(frame(a_mm=53.0), DIMS_ONLY)
    assert compensating is not None and uncompensated is not None
    assert compensating.components["optical_centration"] == 1.0
    assert uncompensated.components["optical_centration"] < 1.0


def test_bridge_error_is_asymmetric_narrow_worse_than_wide() -> None:
    narrow = score_frame(frame(dbl_mm=16.5), DIMS_ONLY)
    wide = score_frame(frame(dbl_mm=19.5), DIMS_ONLY)
    assert narrow is not None and wide is not None
    assert narrow.components["bridge_fit"] < wide.components["bridge_fit"]


def test_adjustable_pads_soften_bridge_error() -> None:
    fixed = score_frame(frame(dbl_mm=19.5), DIMS_ONLY)
    adjustable = score_frame(frame(dbl_mm=19.5, nose_pads="adjustable"), DIMS_ONLY)
    assert fixed is not None and adjustable is not None
    assert adjustable.components["bridge_fit"] > fixed.components["bridge_fit"]


def test_deep_lens_near_cheeks_penalized_and_flagged() -> None:
    # sample cheek clearance ~5.75mm; B=44 at pupil ratio ~0.545 puts ~20mm below the pupil
    ctx = measured_ctx()
    deep = score_frame(frame(b_mm=44.0), ctx)
    shallow = score_frame(frame(b_mm=36.0), ctx)
    assert deep is not None and shallow is not None
    assert deep.components["lens_height"] < shallow.components["lens_height"]
    assert any("cheek" in f.lower() for f in deep.flags)


def test_width_fit_uses_measured_temple_width() -> None:
    # sample temple width 138; front(=2A+DBL+12) for A=52 -> 134, overhang +4 (mild pressure)
    narrow_frame = score_frame(frame(a_mm=48.0, dbl_mm=18.0), measured_ctx())
    wide_frame = score_frame(frame(a_mm=52.0), measured_ctx())
    assert narrow_frame is not None and wide_frame is not None
    # A=48 -> front 126, overhang +12 -> real pressure, worse than A=52's +4
    assert narrow_frame.components["width_fit"] < wide_frame.components["width_fit"]


def test_shape_affinity_prefers_angular_frames_on_soft_jaw() -> None:
    # sample jaw ratio 0.74 < 0.78 -> non-angular face -> angular frames contrast well
    rect = score_frame(frame(shape="rectangle"), measured_ctx())
    round_ = score_frame(frame(shape="round"), measured_ctx())
    assert rect is not None and round_ is not None
    assert rect.components["shape_affinity"] > round_.components["shape_affinity"]


def test_dims_only_context_degrades_to_neutral_components() -> None:
    match = score_frame(frame(), DIMS_ONLY)
    assert match is not None
    assert match.components["wrap_harmony"] == NEUTRAL
    assert match.components["shape_affinity"] == NEUTRAL
    # dimension-driven components still judge fully
    assert match.components["optical_centration"] == 1.0
    assert match.components["bridge_fit"] == 1.0


def test_low_bridge_preference_bonus_and_flags() -> None:
    ctx = MatchContext(targets=TARGETS, prefer_low_bridge=True)
    fixed = score_frame(frame(frame_id="fixed", dbl_mm=19.0), ctx)
    adjustable = score_frame(frame(frame_id="adj", dbl_mm=19.0, nose_pads="adjustable"), ctx)
    assert fixed is not None and adjustable is not None
    assert adjustable.components["bridge_fit"] > fixed.components["bridge_fit"]
    assert any("Low-bridge friendly" in f for f in adjustable.flags)
    assert any("may slide" in f for f in fixed.flags)


def test_bonus_cannot_saturate_ranking() -> None:
    # Regression (v1): an additive bonus clamped both frames to the max score, letting
    # the WORSE frame win on the alphabetical tie-break.
    catalog = [
        frame(frame_id="A-1", a_mm=53.0, nose_pads="adjustable"),
        frame(frame_id="Z-9", nose_pads="adjustable"),  # strictly better dims
    ]
    ctx = MatchContext(targets=TARGETS, prefer_low_bridge=True)
    matches = match_frames(catalog, ctx)
    assert [m.frame.frame_id for m in matches] == ["Z-9", "A-1"]
    assert matches[0].fit_score > matches[1].fit_score


def test_ranking_deterministic_and_limited() -> None:
    catalog = [frame(frame_id=f"F-{i}", a_mm=52.0 + i * 0.5) for i in range(10)]
    matches = match_frames(catalog, DIMS_ONLY, limit=3)
    assert len(matches) == 3
    scores = [m.fit_score for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_deltas_are_frame_minus_target() -> None:
    match = score_frame(frame(a_mm=54.0, dbl_mm=17.0), DIMS_ONLY)
    assert match is not None
    assert match.deltas["a_mm"] == 2.0
    assert match.deltas["dbl_mm"] == -1.0


# --- API flow ---------------------------------------------------------------------------------


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
        # hard filter: ±2mm fixed pads, ±3mm adjustable
        tol = 3.0 if m["frame"]["nose_pads"] == "adjustable" else 2.0
        assert abs(m["deltas"]["dbl_mm"]) <= tol
        assert set(m["components"]) == {
            "optical_centration",
            "bridge_fit",
            "width_fit",
            "lens_height",
            "temple_fit",
            "wrap_harmony",
            "weight_slip",
            "shape_affinity",
        }
        # measurement-driven context: no neutral placeholders
        assert (
            m["components"]["shape_affinity"] != NEUTRAL
            or m["components"]["wrap_harmony"] != NEUTRAL
        )

    # rate the top match — the learning-loop label
    top = body["matches"][0]
    rated = client.post(
        "/api/v1/frames/ratings",
        json={
            "recommendation_id": rec["recommendation_id"],
            "frame_id": top["frame"]["frame_id"],
            "rating": 4,
            "fit_score": top["fit_score"],
            "components": top["components"],
        },
    )
    assert rated.status_code == 201, rated.text
    assert rated.json()["rating_id"]


def test_match_with_inline_targets(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/frames/match",
        json={"targets": json.loads(TARGETS.model_dump_json())},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["matches"]) > 0


def test_match_validation_and_unknown_recommendation(client: TestClient) -> None:
    assert client.post("/api/v1/frames/match", json={}).status_code == 422
    resp = client.post("/api/v1/frames/match", json={"recommendation_id": "nope"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_rating_validation_and_unknowns(client: TestClient) -> None:
    scan = client.post("/api/v1/scan", files=jpeg_upload_files(6)).json()
    rec = client.post(
        "/api/v1/recommendations", json={"pd_mm": 63.0, "scan_id": scan["scan_id"]}
    ).json()
    ok = {"recommendation_id": rec["recommendation_id"], "frame_id": "GF-001", "rating": 5}
    assert client.post("/api/v1/frames/ratings", json={**ok, "rating": 0}).status_code == 422
    assert client.post("/api/v1/frames/ratings", json={**ok, "rating": 6}).status_code == 422
    missing_rec = client.post("/api/v1/frames/ratings", json={**ok, "recommendation_id": "nope"})
    assert missing_rec.status_code == 404
    missing_frame = client.post("/api/v1/frames/ratings", json={**ok, "frame_id": "GF-999"})
    assert missing_frame.status_code == 404
    assert client.post("/api/v1/frames/ratings", json=ok).status_code == 201


def test_match_uses_low_bridge_preference_from_measurements(make_app) -> None:
    # The synthetic fixture has a HIGH crest (14mm) -> no low-bridge preference expected.
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


def test_scores_still_ordered_far_beyond_tolerance_spans() -> None:
    # Regression: small faces put the WHOLE catalog past the centration/width spans;
    # a hard floor at 0.0 tied every frame and erased 35% of the ranking signal.
    less_off = score_frame(frame(a_mm=64.0, dbl_mm=17.0), DIMS_ONLY)  # 5.5mm/lens dec
    more_off = score_frame(frame(a_mm=67.0, dbl_mm=17.0), DIMS_ONLY)  # 7.0mm/lens dec
    assert less_off is not None and more_off is not None
    c1 = less_off.components["optical_centration"]
    c2 = more_off.components["optical_centration"]
    assert 0.0 < c2 < c1  # both beyond the span, still strictly ordered


def test_declared_zero_wrap_is_respected() -> None:
    # Regression: `default_wrap_deg or fallback` treated an explicit 0.0deg (flat
    # front) as unset and scored it as the 5deg default.
    flat = score_frame(frame(default_wrap_deg=0.0), measured_ctx())  # target 6deg
    unset = score_frame(frame(), measured_ctx())
    assert flat is not None and unset is not None
    assert flat.components["wrap_harmony"] == 0.0  # |0-6|/6 -> fully off-target
    assert unset.components["wrap_harmony"] > flat.components["wrap_harmony"]
