"""Shared, fully-populated schema builders for tests.

One home for FaceMeasurements/Recommendation sample objects so schema changes touch a
single file. Importable from training/tests too (pyproject adds backend/tests to
pytest's pythonpath). Override any field via ``model_copy(update=...)`` on the result
or the keyword passthroughs below.
"""

from glassfit.schemas import (
    AsWorn,
    BehindEar,
    BridgeWidths,
    Comfort,
    FaceMeasurements,
    FrameDims,
    MeasurementQuality,
    NosePads,
    Optics,
    PerSide,
    Recommendation,
    RuleTrace,
    Temples,
)


def side(right: float, left: float | None = None) -> PerSide:
    return PerSide(right=right, left=right if left is None else left)


def sample_measurements(**overrides) -> FaceMeasurements:
    base = FaceMeasurements(
        mm_per_unit=0.25,
        pd_binocular_mm=63.0,
        pd_monocular_mm=side(31.4, 31.6),
        bridge=BridgeWidths(at_crest_mm=16.0, below_10mm_mm=18.5, below_15mm_mm=20.0),
        bridge_crest_height_mm=8.0,
        zygoma_width_mm=132.0,
        temple_width_mm=138.0,
        face_length_mm=142.0,
        jaw_width_mm=98.0,
        face_wrap_radius_mm=95.0,
        cheek_clearance_mm=side(6.0, 5.5),
        hinge_to_ear_mm=side(98.0, 99.0),
        ear_height_asymmetry_mm=1.5,
        behind_ear={
            "right": BehindEar(drop_mm=38.0, angle_deg=42.0),
            "left": BehindEar(drop_mm=39.0, angle_deg=41.0),
        },
        vertex_estimate_mm=12.5,
        canthal_tilt_deg=side(4.0, 3.5),
        pupil_height_ratio=side(0.55, 0.54),
        quality=MeasurementQuality(low_confidence_fields=["hinge_to_ear_mm"]),
    )
    return base.model_copy(update=overrides) if overrides else base


def sample_recommendation(rec_id: str = "rec-1", **overrides) -> Recommendation:
    base = Recommendation(
        recommendation_id=rec_id,
        ruleset_version="rules-1.0.0",
        frame=FrameDims(a_mm=52.0, b_mm=38.0, dbl_mm=18.0, ed_mm=55.0, temple_length_mm=145.0),
        as_worn=AsWorn(pantoscopic_deg=6.5, face_form_deg=6.0, vertex_mm=13.0),
        optics=Optics(
            pd_monocular_mm=side(31.4, 31.6),
            oc_height_mm=side(22.0, 22.5),
            inset_mm=side(2.0, 2.0),
        ),
        nose_pads=NosePads(size="M", splay_deg=30.0, flare_deg=10.0, drop_mm=3.0),
        temples=Temples(
            bend_point_mm_from_hinge=side(98.0, 99.0),
            tip_angle_deg=45.0,
            raise_mm=side(0.0, 1.0),
        ),
        comfort=Comfort(predicted_slip=0.2, nose_pressure=0.3, temple_pressure=0.25),
        notes=["Pantoscopic tilt kept within 5-8 deg."],
        rule_trace={
            "as_worn.pantoscopic_deg": RuleTrace(
                rule_id="panto_default",
                inputs={"canthal_tilt": 4.0},
                raw_value=6.5,
                clamped=False,
            )
        },
    )
    return base.model_copy(update=overrides) if overrides else base
