"""Hand-computed cases for the pure geometry helpers."""

import numpy as np
import pytest

from glassfit.measure.geometry import (
    angle_to_horizontal_deg,
    dist,
    fit_circle_radius,
    midpoint,
    polyline_station,
)


def test_dist_345() -> None:
    assert dist((0, 0), (3, 4)) == pytest.approx(5.0)
    assert dist((1, 2, 3), (1, 2, 3)) == 0.0


def test_midpoint() -> None:
    assert midpoint((0, 0), (4, 6)) == pytest.approx([2.0, 3.0])


def test_angle_to_horizontal() -> None:
    assert angle_to_horizontal_deg((2, 0)) == pytest.approx(0.0)
    assert angle_to_horizontal_deg((1, 1)) == pytest.approx(45.0)
    # mirror-invariant in x, sign follows y
    assert angle_to_horizontal_deg((-1, 1)) == pytest.approx(45.0)
    assert angle_to_horizontal_deg((1, -1)) == pytest.approx(-45.0)


def test_fit_circle_radius_exact() -> None:
    unit = [(0, 1), (1, 0), (0, -1)]
    assert fit_circle_radius(unit) == pytest.approx(1.0)
    scaled = np.array(unit) * 7.5 + np.array([3.0, -2.0])
    assert fit_circle_radius(scaled) == pytest.approx(7.5)


def test_fit_circle_collinear_reports_flat_not_tiny() -> None:
    # A perfectly flat face profile must read as a HUGE radius (flat arc); the
    # unguarded lstsq solution used to return an arbitrary small one, which would
    # masquerade as a maximally curved face and drive maximum wrap.
    from glassfit.measure.geometry import FLAT_RADIUS

    assert fit_circle_radius([(-65.0, 55.0), (0.0, 55.0), (65.0, 55.0)]) == FLAT_RADIUS
    # near-collinear (sub-micron sag over 130mm) also snaps to flat
    assert fit_circle_radius([(-65.0, 55.0), (0.0, 55.0000001), (65.0, 55.0)]) == FLAT_RADIUS
    # a genuinely curved face is untouched by the guard
    assert fit_circle_radius([(-65.0, 55.0), (0.0, 0.0), (65.0, 55.0)]) < 1000.0


def test_polyline_station() -> None:
    chain = [(0, 0), (10, 0), (10, 10)]
    assert polyline_station(chain, (5, 1)) == pytest.approx(5.0)
    assert polyline_station(chain, (11, 5)) == pytest.approx(15.0)
