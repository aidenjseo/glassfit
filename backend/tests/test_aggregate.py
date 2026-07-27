"""Tests for glassfit.vision.aggregate (pure numpy, no mediapipe)."""

import numpy as np
import pytest

from glassfit.vision.aggregate import aggregate_landmarks, procrustes_align

WIDTH, HEIGHT = 640, 480
N_LANDMARKS = 478


def _base_landmarks() -> np.ndarray:
    """Deterministic normalized (478, 3) landmark cloud."""
    rng = np.random.default_rng(42)
    pts = rng.uniform([0.25, 0.2, -0.05], [0.75, 0.8, 0.05], size=(N_LANDMARKS, 3))
    return pts


def _jittered(base: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base + rng.normal(0.0, sigma, size=base.shape)


def test_median_recovers_original_within_epsilon() -> None:
    base = _base_landmarks()
    sigma = 0.0005  # ~0.3 px at 640 wide
    sets = [_jittered(base, sigma, seed=i) for i in range(6)]

    canonical, dispersion = aggregate_landmarks(sets, WIDTH, HEIGHT)

    assert canonical.shape == (N_LANDMARKS, 3)
    np.testing.assert_allclose(canonical, base, atol=5 * sigma)
    assert dispersion >= 0.0


def test_gross_outlier_frame_does_not_shift_median() -> None:
    base = _base_landmarks()
    sigma = 0.0005
    clean_sets = [_jittered(base, sigma, seed=i) for i in range(5)]
    outlier = _jittered(base, 0.05, seed=99)  # ~32 px of structural noise

    clean_median, _ = aggregate_landmarks(clean_sets, WIDTH, HEIGHT)
    dirty_median, _ = aggregate_landmarks([*clean_sets, outlier], WIDTH, HEIGHT)

    shift_px = np.abs((dirty_median - clean_median) * [WIDTH, HEIGHT, WIDTH]).max()
    assert shift_px < 1.0  # median stays inside the tight cluster
    np.testing.assert_allclose(dirty_median, base, atol=5 * sigma)


def test_dispersion_grows_with_jitter() -> None:
    base = _base_landmarks()
    low_sets = [_jittered(base, 0.0005, seed=i) for i in range(6)]
    high_sets = [_jittered(base, 0.005, seed=100 + i) for i in range(6)]

    _, low_dispersion = aggregate_landmarks(low_sets, WIDTH, HEIGHT)
    _, high_dispersion = aggregate_landmarks(high_sets, WIDTH, HEIGHT)

    assert 0.0 < low_dispersion < high_dispersion


def test_single_set_is_identity_with_zero_dispersion() -> None:
    base = _base_landmarks()
    canonical, dispersion = aggregate_landmarks([base], WIDTH, HEIGHT)
    np.testing.assert_allclose(canonical, base, atol=1e-12)
    assert dispersion == 0.0


def test_procrustes_removes_rotation_translation_scale() -> None:
    base_px = _base_landmarks() * [WIDTH, HEIGHT, WIDTH]
    theta = np.deg2rad(25.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    transformed = 1.15 * (base_px @ rotation.T) + np.array([30.0, -12.0, 4.0])

    aligned = procrustes_align(transformed, base_px)
    np.testing.assert_allclose(aligned, base_px, atol=1e-8)


def test_aggregate_transformed_copies_recovers_reference() -> None:
    base = _base_landmarks()
    base_px = base * [WIDTH, HEIGHT, WIDTH]
    theta = np.deg2rad(10.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    moved_px = 0.9 * (base_px @ rotation.T) + np.array([-15.0, 8.0, 0.0])
    moved = moved_px / [WIDTH, HEIGHT, WIDTH]

    canonical, dispersion = aggregate_landmarks([base, moved, base], WIDTH, HEIGHT)
    np.testing.assert_allclose(canonical, base, atol=1e-6)
    assert dispersion < 1e-6


def test_empty_and_mismatched_inputs_raise() -> None:
    base = _base_landmarks()
    with pytest.raises(ValueError):
        aggregate_landmarks([], WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        aggregate_landmarks([base, base[:100]], WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        procrustes_align(base[:10], base[:11])
