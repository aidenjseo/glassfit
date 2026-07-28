"""Feature engineering for GlassFit models.

Ratings model (the frame-match learning loop):
    features = the matcher's own component snapshot (``comp_*``) + fit_score,
    catalog-frame attributes (numeric dims + categorical shape/material/pads),
    and the face measurements (``m_*`` scalars). HistGradientBoosting consumes
    pandas ``category`` dtypes natively and tolerates NaNs (older rating rows
    without a component snapshot simply carry missing values).

Comfort model (worn-fit feedback):
    features = face measurements + the rules engine's own outputs (``rec_*``).
"""

from typing import Any

import pandas as pd

RATING_NUMERIC = [
    "fit_score",
    "frame_a_mm",
    "frame_b_mm",
    "frame_dbl_mm",
    "frame_ed_mm",
    "frame_temple_mm",
    "frame_weight_g",
    "pd_mm",
]
RATING_CATEGORICAL = ["frame_shape", "frame_material", "frame_nose_pads"]
MEASUREMENT_SCALARS = [
    "m_pd_binocular_mm",
    "m_zygoma_width_mm",
    "m_temple_width_mm",
    "m_face_length_mm",
    "m_jaw_width_mm",
    "m_bridge_crest_height_mm",
    "m_vertex_estimate_mm",
    "m_face_wrap_radius_mm",
]


def build_rating_features(frame: Any) -> tuple[Any, list[str]]:
    """Ratings DataFrame -> (X, feature_names). Missing columns become NaN."""
    comp_cols = sorted(c for c in frame.columns if c.startswith("comp_"))
    numeric = comp_cols + RATING_NUMERIC + MEASUREMENT_SCALARS
    x = pd.DataFrame(index=frame.index)
    for col in numeric:
        x[col] = pd.to_numeric(frame.get(col), errors="coerce")
    for col in RATING_CATEGORICAL:
        x[col] = frame.get(col, pd.Series(index=frame.index, dtype=object)).astype("category")
    return x, list(x.columns)


def build_comfort_features(frame: Any) -> tuple[Any, list[str]]:
    """Feedback DataFrame -> (X, feature_names): measurements + rules outputs."""
    numeric = [
        c
        for c in frame.columns
        if (c.startswith("m_") or c.startswith("rec_") or c == "pd_mm")
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    x = frame[sorted(numeric)].copy()
    return x, list(x.columns)
