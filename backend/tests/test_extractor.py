"""Extractor vs synthetic fixtures with known ground-truth geometry."""

import numpy as np
import pytest
from conftest import load_landmark_set, truncate_to_468

from glassfit.measure.extractor import LOW_CONFIDENCE_FIELDS, compute_measurements
from glassfit.schemas import FaceMeasurements, LandmarkSet

NAMES = ["synthetic_average", "synthetic_narrow", "synthetic_wide"]


def _load(name: str) -> tuple[LandmarkSet, float, dict]:
    ls, payload = load_landmark_set(name)
    return ls, payload["pd_mm_ground_truth"], payload["expected"]


@pytest.fixture(scope="module", params=NAMES)
def case(request: pytest.FixtureRequest) -> tuple[FaceMeasurements, dict]:
    ls, pd, expected = _load(request.param)
    return compute_measurements(ls, pd), expected


def test_exact_widths_and_pd(case: tuple[FaceMeasurements, dict]) -> None:
    m, exp = case
    assert m.pd_binocular_mm == pytest.approx(exp["pd_binocular_mm"], abs=1e-6)
    assert m.pd_monocular_mm.right == pytest.approx(exp["pd_monocular_right_mm"], abs=1e-6)
    assert m.pd_monocular_mm.left == pytest.approx(exp["pd_monocular_left_mm"], abs=1e-6)
    assert m.zygoma_width_mm == pytest.approx(exp["zygoma_width_mm"], abs=1e-6)
    assert m.temple_width_mm == pytest.approx(exp["temple_width_mm"], abs=1e-6)
    # 1e-4: fixture coords are serialized at 8 normalized decimals (~3e-6 mm error)
    assert m.face_length_mm == pytest.approx(exp["face_length_mm"], abs=1e-4)
    assert m.jaw_width_mm == pytest.approx(exp["jaw_width_mm"], abs=1e-4)


def test_vertical_and_depth_features(case: tuple[FaceMeasurements, dict]) -> None:
    m, exp = case
    assert m.bridge_crest_height_mm == pytest.approx(exp["bridge_crest_height_mm"], abs=1e-6)
    assert m.cheek_clearance_mm.right == pytest.approx(exp["cheek_clearance_mm"], abs=1e-6)
    assert m.cheek_clearance_mm.left == pytest.approx(exp["cheek_clearance_mm"], abs=1e-6)
    assert m.vertex_estimate_mm == pytest.approx(exp["vertex_estimate_mm"], abs=1e-6)
    assert m.pupil_height_ratio.right == pytest.approx(exp["pupil_height_ratio"], abs=1e-3)
    assert m.canthal_tilt_deg.right == pytest.approx(exp["canthal_tilt_deg"], abs=0.1)
    assert m.canthal_tilt_deg.left == pytest.approx(exp["canthal_tilt_deg"], abs=0.1)


def test_bridge_widths_plausible(case: tuple[FaceMeasurements, dict]) -> None:
    m, _ = case
    # designed sidewall widths are 17-21mm; interpolated stations stay in that band
    for width in (m.bridge.at_crest_mm, m.bridge.below_10mm_mm, m.bridge.below_15mm_mm):
        assert 15.0 <= width <= 23.0


def test_ear_estimates_and_confidence(case: tuple[FaceMeasurements, dict]) -> None:
    m, exp = case
    assert m.hinge_to_ear_mm.right == pytest.approx(exp["hinge_to_ear_mm"], abs=1e-6)
    assert m.ear_height_asymmetry_mm == pytest.approx(exp["ear_height_asymmetry_mm"], abs=1e-6)
    assert m.behind_ear["right"].drop_mm == pytest.approx(exp["behind_ear_drop_mm"], abs=1e-6)
    assert m.face_wrap_radius_mm == pytest.approx(exp["face_wrap_radius_mm"], rel=0.05)
    assert set(LOW_CONFIDENCE_FIELDS) <= set(m.quality.low_confidence_fields)
    assert not m.quality.scale_suspect


def test_mirrored_frame_gives_identical_measurements() -> None:
    ls, pd, _ = _load("synthetic_average")
    m = compute_measurements(ls, pd)
    pts = np.asarray(ls.points, dtype=float)
    pts[:, 0] = 1.0 - pts[:, 0]
    mirrored = LandmarkSet(
        points=[tuple(p) for p in pts], image_width=ls.image_width, image_height=ls.image_height
    )
    mm = compute_measurements(mirrored, pd)
    assert mm.pd_binocular_mm == pytest.approx(m.pd_binocular_mm, abs=1e-6)
    assert mm.zygoma_width_mm == pytest.approx(m.zygoma_width_mm, abs=1e-6)
    assert mm.pd_monocular_mm.right == pytest.approx(m.pd_monocular_mm.right, abs=1e-6)
    assert mm.canthal_tilt_deg.right == pytest.approx(m.canthal_tilt_deg.right, abs=1e-6)


def test_scale_suspect_on_implausible_pd() -> None:
    # wide-face geometry with a tiny PD implies a sub-100mm zygoma -> suspect flag
    ls, _, _ = _load("synthetic_wide")
    m = compute_measurements(ls, 45.0)
    assert m.quality.scale_suspect
    assert any("scale_suspect" in w for w in m.quality.warnings)


def test_fallback_warning_propagates_into_quality() -> None:
    ls, pd, _ = _load("synthetic_average")
    truncated = truncate_to_468(ls)
    m = compute_measurements(truncated, pd)
    assert "pd_scale_fallback_eye_corners" in m.quality.warnings
