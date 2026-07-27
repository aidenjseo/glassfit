"""Hand-computed cases for the pure geometry helpers."""

import numpy as np
import pytest

from glassfit.measure.geometry import (
    angle_to_horizontal_deg,
    dist,
    fit_circle_radius,
    midpoint,
    point_line_distance,
    polyline_point_at,
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


def test_point_line_distance() -> None:
    assert point_line_distance((0, 1), (-5, 0), (5, 0)) == pytest.approx(1.0)
    # degenerate line -> point distance
    assert point_line_distance((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)


def test_fit_circle_radius_exact() -> None:
    unit = [(0, 1), (1, 0), (0, -1)]
    assert fit_circle_radius(unit) == pytest.approx(1.0)
    scaled = np.array(unit) * 7.5 + np.array([3.0, -2.0])
    assert fit_circle_radius(scaled) == pytest.approx(7.5)


def test_polyline_station_and_point_at() -> None:
    chain = [(0, 0), (10, 0), (10, 10)]
    assert polyline_station(chain, (5, 1)) == pytest.approx(5.0)
    assert polyline_station(chain, (11, 5)) == pytest.approx(15.0)
    assert polyline_point_at(chain, 15.0) == pytest.approx([10.0, 5.0])
    assert polyline_point_at(chain, 999.0) == pytest.approx([10.0, 10.0])  # clamped
