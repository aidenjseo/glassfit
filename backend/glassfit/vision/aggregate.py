"""Multi-frame landmark aggregation: Procrustes alignment + robust median. Pure numpy."""

import numpy as np


def procrustes_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity-align ``src`` onto ``dst`` (both ``(N, D)``), returning the transformed copy.

    Least-squares similarity transform (uniform scale + rotation + translation, Umeyama 1991)
    minimizing ``||s * R @ src + t - dst||^2``. No reflections: the rotation determinant is
    forced to +1. Degenerate (zero-variance) sources are returned translated only.
    """
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: src {src.shape} vs dst {dst.shape}")
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = float((src_c**2).sum()) / n
    if var_src < 1e-12:
        return np.broadcast_to(mu_dst, src.shape).copy()

    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(src.shape[1])
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    scale = float(s @ d) / var_src
    return scale * (src_c @ rotation.T) + mu_dst


def aggregate_landmarks(
    sets: list[np.ndarray],
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, float]:
    """Fuse per-frame NORMALIZED landmark sets into one canonical set.

    Each set ``(L, 3)`` is converted to PIXEL space (x*width, y*height, z*width — z is
    x-scaled per the MediaPipe convention), similarity-Procrustes-aligned to the FIRST set
    (translation + rotation + uniform scale), then fused by per-landmark coordinate-wise MEDIAN,
    which is robust to a minority of outlier frames.

    Returns:
        ``(canonical, dispersion_px)`` where ``canonical`` is the ``(L, 3)`` median set converted
        back to NORMALIZED coordinates, and ``dispersion_px`` is the median over landmarks of the
        per-landmark scatter in PIXELS (per-landmark scatter = sqrt(std_x^2 + std_y^2) across the
        aligned frames, i.e. the RMS in-plane radial deviation).

    Raises:
        ValueError: on an empty list or mismatched set shapes.
    """
    if not sets:
        raise ValueError("aggregate_landmarks requires at least one landmark set")

    to_px = np.array([image_width, image_height, image_width], dtype=np.float64)
    reference = np.asarray(sets[0], dtype=np.float64) * to_px
    aligned = [reference]
    for landmark_set in sets[1:]:
        arr = np.asarray(landmark_set, dtype=np.float64)
        if arr.shape != reference.shape:
            raise ValueError(f"shape mismatch: {arr.shape} vs {reference.shape}")
        aligned.append(procrustes_align(arr * to_px, reference))

    stack = np.stack(aligned)  # (F, L, 3) pixels
    median_px = np.median(stack, axis=0)

    std_xy = stack[:, :, :2].std(axis=0)  # (L, 2) per-landmark per-axis std in pixels
    per_landmark_scatter = np.sqrt((std_xy**2).sum(axis=1))  # (L,)
    dispersion_px = float(np.median(per_landmark_scatter))

    return median_px / to_px, dispersion_px
