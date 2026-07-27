"""Canonical MediaPipe FaceLandmarker (478-point) landmark indices.

Indices 0-467 are the face mesh; 468-477 are the 10 iris points added by the landmarker.

Sides here are SUBJECT-anatomical ("right" = the subject's right eye, i.e. OD), per the official
mediapipe `face_mesh_connections.py`. Third-party index tables in the wild contradict each other
(image-space vs anatomical convention) — so NEVER trust side labels blindly at runtime: verify by
x-coordinate (in an unmirrored frame the subject's RIGHT iris center has the SMALLER x).

Confidence notes: entries marked LOW must be calibrated against the canonical face model UV map
(mediapipe/modules/face_geometry/data/) during measurement tuning.
"""

# --- Iris (HIGH confidence: official FACEMESH_*_IRIS) ---
IRIS_CENTER_RIGHT = 468
IRIS_CENTER_LEFT = 473
IRIS_RING_RIGHT = (469, 470, 471, 472)
IRIS_RING_LEFT = (474, 475, 476, 477)

# --- Eyes (HIGH: official FACEMESH_RIGHT_EYE / FACEMESH_LEFT_EYE endpoints) ---
OUTER_CANTHUS_RIGHT = 33
OUTER_CANTHUS_LEFT = 263
INNER_CANTHUS_RIGHT = 133
INNER_CANTHUS_LEFT = 362
# Eyelids (MEDIUM-HIGH: standard eye-aspect-ratio points)
UPPER_LID_RIGHT = 159
LOWER_LID_RIGHT = 145
UPPER_LID_LEFT = 386
LOWER_LID_LEFT = 374

# --- Nose (HIGH for midline chain; tip 1 vs 4 are both apex-region — verify visually) ---
NASION = 168  # sellion region, "midway between eyes"
NOSE_DORSUM = (168, 6, 197, 195, 5)  # midline chain, top -> tip-ward
NOSE_TIP = 1
NOSE_TIP_ALT = 4
NOSE_BASE = 2  # subnasale
NOSTRIL_CORNER_RIGHT = 98
NOSTRIL_CORNER_LEFT = 327
ALAR_BASE_RIGHT = 129  # MEDIUM
ALAR_BASE_LEFT = 358  # MEDIUM
# Bridge sidewall candidate pairs at increasing depth below crest (LOW — calibrate!)
BRIDGE_SIDEWALL_PAIRS = ((193, 417), (122, 351), (47, 277))  # (right, left) per pair

# --- Cheeks / face width ---
MID_CHEEK_RIGHT = 205  # malar (HIGH: tfjs rightCheek)
MID_CHEEK_LEFT = 425
FACE_SIDE_RIGHT = 234  # most-lateral face-oval point, pre-auricular ~ tragus proxy (HIGH)
FACE_SIDE_LEFT = 454
TEMPLE_RIGHT = 127  # face-oval, temple region (HIGH)
TEMPLE_LEFT = 356
JAW_RIGHT = (58, 132)  # gonion-ish (MEDIUM)
JAW_LEFT = (288, 361)

# --- Midline extremes (HIGH: face oval) ---
CHIN = 152
FOREHEAD_TOP = 10

# Face-oval index ring (HIGH: official FACEMESH_FACE_OVAL order)
FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

# NOTE: the mesh has NO ear/tragus landmarks. Every ear-dependent measurement (hinge-to-ear,
# temple length, behind-ear bend, ear-height asymmetry) is an ESTIMATE from FACE_SIDE_* proxies
# plus anthropometric offsets, and must be flagged in MeasurementQuality.low_confidence_fields.
