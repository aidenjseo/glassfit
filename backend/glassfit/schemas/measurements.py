"""Facial measurement schemas (all lengths mm, all angles degrees)."""

from pydantic import BaseModel, Field, model_validator

from .common import LandmarkSet, PerSide


class RxEye(BaseModel):
    sphere: float = Field(0.0, ge=-20, le=20)
    cyl: float = Field(0.0, ge=-10, le=10)
    axis: int = Field(0, ge=0, le=180)


class Rx(BaseModel):
    od: RxEye = RxEye()  # subject right eye
    os: RxEye = RxEye()  # subject left eye


class MeasurementRequest(BaseModel):
    pd_mm: float = Field(ge=45, le=80)
    scan_id: str | None = None
    landmarks: LandmarkSet | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "MeasurementRequest":
        if (self.scan_id is None) == (self.landmarks is None):
            raise ValueError("provide exactly one of scan_id or landmarks")
        return self


class BridgeWidths(BaseModel):
    """Nose-bridge width sampled at three heights below the crest."""

    at_crest_mm: float
    below_10mm_mm: float
    below_15mm_mm: float


class BehindEar(BaseModel):
    drop_mm: float
    angle_deg: float


class MeasurementQuality(BaseModel):
    scale_suspect: bool = False  # implied face size implausible -> "check PD entry"
    low_confidence_fields: list[str] = []
    warnings: list[str] = []


class FaceMeasurements(BaseModel):
    mm_per_unit: float
    pd_binocular_mm: float
    pd_monocular_mm: PerSide  # nasion-split monocular PDs
    bridge: BridgeWidths
    bridge_crest_height_mm: float
    zygoma_width_mm: float
    temple_width_mm: float
    face_length_mm: float  # forehead-top (10) to chin (152) — drives shape affinity
    jaw_width_mm: float  # gonion-to-gonion (58<->288) — drives shape affinity
    face_wrap_radius_mm: float
    cheek_clearance_mm: PerSide
    hinge_to_ear_mm: PerSide  # ESTIMATE — mesh has no ear landmarks (low confidence)
    # SIGN CONVENTION (extractor + rules engine agree): > 0 means the subject's RIGHT
    # ear sits HIGHER than the left; the higher-ear temple gets raised to level the frame.
    ear_height_asymmetry_mm: float  # ESTIMATE (low confidence)
    behind_ear: dict[str, BehindEar]  # {"right": ..., "left": ...} ESTIMATE (low confidence)
    vertex_estimate_mm: float
    canthal_tilt_deg: PerSide
    pupil_height_ratio: PerSide  # pupil vertical position within palpebral aperture (0=bottom lid)
    quality: MeasurementQuality = MeasurementQuality()
