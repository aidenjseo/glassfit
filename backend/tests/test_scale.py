"""Scale anchoring and runtime side resolution."""

import json
from pathlib import Path

import numpy as np
import pytest

from glassfit.measure.scale import (
    PD_SCALE_FALLBACK_WARNING,
    mm_per_unit,
    resolve_sides,
    to_unit_space,
)
from glassfit.schemas import LandmarkSet

FIXTURE = Path(__file__).parent / "fixtures" / "landmarks" / "synthetic_average.json"


def _fixture_landmarks() -> tuple[LandmarkSet, float]:
    data = json.loads(FIXTURE.read_text())
    return LandmarkSet(**data["landmark_set"]), data["pd_mm_ground_truth"]


def test_to_unit_space_is_isotropic_pixels() -> None:
    ls = LandmarkSet(
        points=[(0.5, 0.5, 0.1)] * 468,
        image_width=1280,
        image_height=720,
    )
    pts = to_unit_space(ls)
    assert pts.shape == (468, 3)
    assert pts[0] == pytest.approx([640.0, 360.0, 128.0])


def test_mm_per_unit_iris_exact() -> None:
    ls, pd = _fixture_landmarks()
    pts = to_unit_space(ls)
    scale, warnings = mm_per_unit(pd, pts, has_iris=True)
    # fixture is built at 4 px/mm -> 0.25 mm per pixel unit
    assert scale == pytest.approx(0.25, rel=1e-9)
    assert warnings == []


def test_mm_per_unit_fallback_on_468_points() -> None:
    ls, pd = _fixture_landmarks()
    truncated = LandmarkSet(
        points=ls.points[:468], image_width=ls.image_width, image_height=ls.image_height
    )
    pts = to_unit_space(truncated)
    scale, warnings = mm_per_unit(pd, pts, has_iris=truncated.has_iris)
    assert scale > 0
    assert PD_SCALE_FALLBACK_WARNING in warnings


def test_resolve_sides_canonical_and_mirrored() -> None:
    ls, _ = _fixture_landmarks()
    pts = to_unit_space(ls)
    sides = resolve_sides(pts)
    assert not sides.flipped
    assert sides.right.iris_center == 468

    mirrored = np.asarray(ls.points, dtype=float)
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    flipped = resolve_sides(mirrored)
    assert flipped.flipped
    assert flipped.right.iris_center == 473
