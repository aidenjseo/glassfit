"""Generate geometrically consistent synthetic 478-point landmark fixtures.

Key landmarks are placed in MILLIMETER space (nasion at the origin, x toward the
subject's LEFT, y DOWN, z AWAY from the camera) at positions derived from target
ground-truth measurements, then projected to normalized image coordinates for a
1280x720 frame at K_PX_PER_MM. Because the extractor's scale factor is anchored on
the iris-center distance, every designed distance is recovered exactly.

Run once to (re)write the JSON fixtures:
    uv run python backend/tests/fixtures/make_synthetic.py
"""

import json
import math
from pathlib import Path

import numpy as np

from glassfit.measure import indices as idx

OUT_DIR = Path(__file__).parent / "landmarks"
IMAGE_W, IMAGE_H = 1280, 720
K_PX_PER_MM = 4.0
CENTER_PX = (640.0, 310.0)  # nasion position in the frame


def _key_positions_mm(pd: float, zygoma: float, temple: float) -> dict[int, tuple]:
    """Millimeter-space positions for every landmark index the extractor uses."""
    hp, hz, ht = pd / 2.0, zygoma / 2.0, temple / 2.0
    pos: dict[int, tuple] = {
        idx.NASION: (0, 0, 0),
        idx.IRIS_CENTER_RIGHT: (-hp, 2, 12),
        idx.IRIS_CENTER_LEFT: (hp, 2, 12),
        idx.OUTER_CANTHUS_RIGHT: (-hp - 12.5, 2.5, 18),
        idx.OUTER_CANTHUS_LEFT: (hp + 12.5, 2.5, 18),
        idx.INNER_CANTHUS_RIGHT: (-hp + 14.5, 3, 14),
        idx.INNER_CANTHUS_LEFT: (hp - 14.5, 3, 14),
        idx.UPPER_LID_RIGHT: (-hp, -3.5, 12),
        idx.UPPER_LID_LEFT: (hp, -3.5, 12),
        idx.LOWER_LID_RIGHT: (-hp, 8.5, 12),
        idx.LOWER_LID_LEFT: (hp, 8.5, 12),
        # nose dorsum chain (168 already placed) + tip/base
        6: (0, 7, 1),
        197: (0, 14, 3),
        195: (0, 21, 6),
        5: (0, 27, 9),
        idx.NOSE_TIP: (0, 33, -10),
        idx.NOSE_TIP_ALT: (0, 31, -9),
        idx.NOSE_BASE: (0, 38, -2),
        idx.NOSTRIL_CORNER_RIGHT: (-13, 34, -2),
        idx.NOSTRIL_CORNER_LEFT: (13, 34, -2),
        idx.ALAR_BASE_RIGHT: (-16, 30, 2),
        idx.ALAR_BASE_LEFT: (16, 30, 2),
        idx.MID_CHEEK_RIGHT: (-28, 30, 18),
        idx.MID_CHEEK_LEFT: (28, 30, 18),
        idx.FACE_SIDE_RIGHT: (-hz, 25, 55),
        idx.FACE_SIDE_LEFT: (hz, 25, 55),
        idx.TEMPLE_RIGHT: (-ht, -5, 45),
        idx.TEMPLE_LEFT: (ht, -5, 45),
        idx.JAW_RIGHT[0]: (-48, 60, 40),
        idx.JAW_LEFT[0]: (48, 60, 40),
        idx.JAW_RIGHT[1]: (-45, 65, 35),
        idx.JAW_LEFT[1]: (45, 65, 35),
        idx.CHIN: (0, 85, 5),
        idx.FOREHEAD_TOP: (0, -55, 10),
    }
    # bridge sidewall pairs (right, left) at widths 18 / 17 / 21 mm, increasing depth
    (r1, l1), (r2, l2), (r3, l3) = idx.BRIDGE_SIDEWALL_PAIRS
    pos[r1], pos[l1] = (-9, 6, 2), (9, 6, 2)
    pos[r2], pos[l2] = (-8.5, 12, 4), (8.5, 12, 4)
    pos[r3], pos[l3] = (-10.5, 20, 7), (10.5, 20, 7)
    return pos


def _expected(pd: float, zygoma: float, temple: float) -> dict:
    hz = zygoma / 2.0
    wrap_radius = (hz * hz + 55.0 * 55.0) / (2.0 * 55.0)
    canthal = math.degrees(math.atan2(0.5, 27.0 - (12.5 - 14.5)))  # outer 0.5mm above inner
    return {
        "pd_binocular_mm": pd,
        "pd_monocular_right_mm": pd / 2.0,
        "pd_monocular_left_mm": pd / 2.0,
        "zygoma_width_mm": zygoma,
        "temple_width_mm": temple,
        "bridge_crest_height_mm": 14.0,
        "cheek_clearance_mm": 21.5,
        "pupil_height_ratio": (8.5 - 2.0) / 12.0,
        "vertex_estimate_mm": 12.0,
        "canthal_tilt_deg": canthal,
        "hinge_to_ear_mm": 55.0 - 18.0 + 17.5,
        "ear_height_asymmetry_mm": 0.0,
        "face_wrap_radius_mm": wrap_radius,
        "behind_ear_drop_mm": 0.4 * (60.0 - 25.0),
    }


def build(pd: float, zygoma: float, temple: float, name: str) -> Path:
    rng = np.random.default_rng(42)
    # Filler: a deterministic cloud roughly on the face, overwritten for used indices.
    pts_mm = np.column_stack(
        [
            rng.uniform(-55, 55, 478),
            rng.uniform(-50, 80, 478),
            rng.uniform(0, 50, 478),
        ]
    )
    for i, p in _key_positions_mm(pd, zygoma, temple).items():
        pts_mm[i] = p

    cx, cy = CENTER_PX
    norm = np.column_stack(
        [
            (cx + pts_mm[:, 0] * K_PX_PER_MM) / IMAGE_W,
            (cy + pts_mm[:, 1] * K_PX_PER_MM) / IMAGE_H,
            (pts_mm[:, 2] * K_PX_PER_MM) / IMAGE_W,
        ]
    )
    payload = {
        "pd_mm_ground_truth": pd,
        "expected": _expected(pd, zygoma, temple),
        "landmark_set": {
            "points": [[round(float(v), 8) for v in row] for row in norm],
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(payload) + "\n")
    return out


def main() -> None:
    for pd, zygoma, temple, name in [
        (63.0, 130.0, 118.0, "synthetic_average"),
        (58.0, 118.0, 108.0, "synthetic_narrow"),
        (68.0, 145.0, 132.0, "synthetic_wide"),
    ]:
        print("wrote", build(pd, zygoma, temple, name))


if __name__ == "__main__":
    main()
