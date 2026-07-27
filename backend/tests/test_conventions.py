"""Cross-module convention locks.

The ear-asymmetry sign convention once diverged between the extractor and the rules
engine (two cancelling sign errors produced a correct adjustment with a wrong
explanation). This test drives the REAL pipeline — perturbed landmarks ->
compute_measurements -> recommend — so the modules can never silently disagree again.
"""

import pytest
from conftest import load_landmark_fixture

from glassfit.measure import indices as idx
from glassfit.measure.extractor import compute_measurements
from glassfit.rules.engine import recommend
from glassfit.rules.params import load_rule_params
from glassfit.schemas import LandmarkSet


def _landmarks_with_right_ear_higher(delta_mm: float = 3.0) -> LandmarkSet:
    data = load_landmark_fixture("synthetic_average")["landmark_set"]
    points = [list(p) for p in data["points"]]
    # Raise the subject-RIGHT pre-auricular proxy (234): smaller y = higher on screen.
    # Fixture scale: 4 px/mm on a 720px-high frame.
    points[idx.FACE_SIDE_RIGHT][1] -= (delta_mm * 4.0) / data["image_height"]
    return LandmarkSet(
        points=[tuple(p) for p in points],
        image_width=data["image_width"],
        image_height=data["image_height"],
    )


def test_right_ear_higher_flows_to_right_temple_raise() -> None:
    measurements = compute_measurements(_landmarks_with_right_ear_higher(3.0), 63.0)
    # Extractor convention: positive = subject's RIGHT ear sits higher.
    assert measurements.ear_height_asymmetry_mm == pytest.approx(3.0, abs=0.01)

    params = load_rule_params()
    rec = recommend(measurements, None, None, params)
    # Engine physics: raising a temple LOWERS that frame side, so the higher-ear
    # (right) temple is the one raised — and the note must say "higher".
    assert rec.temples.raise_mm.right > 0
    assert rec.temples.raise_mm.left == 0.0
    note = next(n for n in rec.notes if "temple" in n and "level" in n)
    assert "right" in note and "sits higher" in note
