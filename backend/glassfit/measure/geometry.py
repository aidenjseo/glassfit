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
    "point_line_distance",
    "polyline_arclengths",
    "polyline_point_at",
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


def _as3(v: np.ndarray) -> np.ndarray:
    """Pad a 2-D vector to 3-D (z=0); pass 3-D through unchanged."""
    if v.shape[-1] == 3:
        return v
    out = np.zeros(3)
    out[: v.shape[-1]] = v
    return out


def point_line_distance(p: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """Distance from point ``p`` to the INFINITE line through ``a`` and ``b`` (2-D or 3-D).

    A degenerate line (a == b) falls back to the point-to-point distance.
    """
    p_, a_, b_ = (np.asarray(x, dtype=float) for x in (p, a, b))
    d = b_ - a_
    n = float(np.linalg.norm(d))
    if n == 0.0:
        return dist(a_, p_)
    cross = np.cross(_as3(d), _as3(p_ - a_))
    return float(np.linalg.norm(cross) / n)


def fit_circle(points: ArrayLike) -> tuple[np.ndarray, float]:
    """Least-squares (Kasa) circle fit through 3+ 2-D points -> (center, radius).

    Exact for 3 non-collinear points, algebraic least-squares for more. Collinear input
    yields a very large radius (flat arc) rather than raising.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("fit_circle expects an (N>=3, 2) array of 2-D points")
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    center = np.array([sol[0] / 2.0, sol[1] / 2.0])
    r_sq = float(sol[2] + center[0] ** 2 + center[1] ** 2)
    return center, float(np.sqrt(max(r_sq, 0.0)))


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


def polyline_point_at(chain: ArrayLike, station: float) -> np.ndarray:
    """Point at arc length ``station`` along a polyline (clamped to the chain's ends)."""
    chain_ = np.asarray(chain, dtype=float)
    if len(chain_) < 2:
        return chain_[0].copy()
    cum = polyline_arclengths(chain_)
    s = float(np.clip(station, 0.0, cum[-1]))
    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(chain_) - 2)
    i = max(i, 0)
    seg_len = cum[i + 1] - cum[i]
    t = 0.0 if seg_len == 0.0 else (s - cum[i]) / seg_len
    return chain_[i] + t * (chain_[i + 1] - chain_[i])
