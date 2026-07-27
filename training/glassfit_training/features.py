"""Phase 3 stub — feature engineering for the residual model.

Planned feature vector per training row (all from ``dataset.load_training_frame``):
- ``m_*`` measurement columns (mm/deg scalars; PerSide fields as _right/_left pairs)
- ``rec_*`` rules-engine outputs (the priors the model corrects)
- ``worn_*`` dims of the frame actually worn (falls back to the recommended dims)
- categorical: lens intent, frame tags, nose-pad type; ``user_id`` for personalization
"""

from typing import Any


def build_features(frame: Any) -> Any:
    """DataFrame -> (X, feature_names). Not implemented until Phase 3."""
    raise NotImplementedError("Phase 3")
