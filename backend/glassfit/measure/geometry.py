"""Pure geometric helpers on numpy arrays.

Unit-agnostic: lengths come back in whatever units the inputs use; angles are always
degrees. Points are 1-D arrays (2-D or 3-D accepted) unless a function documents
otherwise. No imports beyond numpy — this module must stay mediapipe/fastapi-free.
"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "angle_to_horizontal_deg",
    "dist",
    "fit_circle",
    "fit_circle_radius",
    "midpoint",
    "polyline_arclengths",
    "polyline_station",
]


def dist(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance between two points of equal dimension."""
    return float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))


def midpoint(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Midpoint of two points."""
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def angle_to_horizontal_deg(v: ArrayLike) -> float:
    """Signed angle in degrees between a vector and the horizontal (+x) axis.

    Only the first two components are used. The sign of the x-component is ignored, so a
    vector and its horizontal mirror report the same angle; the result lies in [-90, 90]
    and is positive when the vector points toward +y. NOTE: with image coordinates
    (y grows downward) a positive result means the vector points DOWN — negate for an
    "upward tilt is positive" convention.
    """
    v_ = np.asarray(v, dtype=float)
    return float(np.degrees(np.arctan2(v_[1], abs(float(v_[0])))))


# Radius reported for (near-)collinear input: effectively "flat", but finite so it
# survives JSON serialization and downstream clamps.
FLAT_RADIUS = 1.0e6


def fit_circle(points: ArrayLike) -> tuple[np.ndarray, float]:
    """Least-squares (Kasa) circle fit through 3+ 2-D points -> (center, radius).

    Exact for 3 non-collinear points, algebraic least-squares for more. (Near-)collinear
    input yields ``FLAT_RADIUS`` (a flat arc) rather than raising — without this guard
    the ill-conditioned lstsq solution can return an arbitrary SMALL radius, which for a
    face-wrap fit would masquerade as a maximally curved face.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("fit_circle expects an (N>=3, 2) array of 2-D points")
    # Degeneracy check: perpendicular spread of the points around their principal axis.
    centered = pts - pts.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    span = float(singular_values[0])
    perpendicular = float(singular_values[-1])
    if span == 0.0 or perpendicular / span < 1.0e-6:
        return pts.mean(axis=0), FLAT_RADIUS
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    center = np.array([sol[0] / 2.0, sol[1] / 2.0])
    r_sq = float(sol[2] + center[0] ** 2 + center[1] ** 2)
    return center, min(float(np.sqrt(max(r_sq, 0.0))), FLAT_RADIUS)


def fit_circle_radius(points: ArrayLike) -> float:
    """Radius of the least-squares circle through 3+ 2-D points."""
    return fit_circle(points)[1]


def polyline_arclengths(chain: ArrayLike) -> np.ndarray:
    """Cumulative arc length at every vertex of a polyline (same length as ``chain``)."""
    chain_ = np.asarray(chain, dtype=float)
    if len(chain_) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(chain_, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_station(chain: ArrayLike, p: ArrayLike) -> float:
    """Arc-length position along ``chain`` of the point on the polyline closest to ``p``.

    Projects ``p`` onto every segment (clamped to segment ends) and returns the cumulative
    arc length of the overall closest projection. 0 for a single-vertex chain.
    """
    chain_ = np.asarray(chain, dtype=float)
    p_ = np.asarray(p, dtype=float)
    if len(chain_) < 2:
        return 0.0
    cum = polyline_arclengths(chain_)
    best_err = np.inf
    best_station = 0.0
    for i in range(len(chain_) - 1):
        a, b = chain_[i], chain_[i + 1]
        d = b - a
        length_sq = float(np.dot(d, d))
        t = 0.0 if length_sq == 0.0 else float(np.clip(np.dot(p_ - a, d) / length_sq, 0.0, 1.0))
        err = float(np.linalg.norm(p_ - (a + t * d)))
        if err < best_err:
            best_err = err
            best_station = float(cum[i] + t * np.sqrt(length_sq))
    return best_station
