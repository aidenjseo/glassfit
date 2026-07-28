"""Rules-engine tests.

Self-contained: FaceMeasurements are built as literals here (no import of the measure module).
"""

import re
from itertools import pairwise
from pathlib import Path

import pytest

from glassfit.catalog.match import EMPIRICAL_JAW_RATIO, EMPIRICAL_LENGTH_RATIO
from glassfit.rules.engine import recommend
from glassfit.rules.params import load_rule_params
from glassfit.schemas import BehindEar, BridgeWidths, FaceMeasurements, PerSide, Rx, RxEye

RULES_PATH = Path(__file__).resolve().parents[1] / "glassfit" / "rules" / "defaults.yaml"

NUMBER_WITH_UNIT = re.compile(r"\d+(\.\d+)?\s?(mm|deg|D)\b")


@pytest.fixture(scope="module")
def params():
    return load_rule_params(RULES_PATH)


def make_measurements(
    *,
    pd_mm: float = 63.0,
    zygoma_mm: float = 130.0,
    temple_width_mm: float = 136.0,
    crest_mm: float = 8.0,
    mid_bridge_mm: float = 17.0,
    wrap_radius_mm: float = 95.0,
    hinge_right_mm: float = 98.0,
    hinge_left_mm: float = 100.0,
    ear_asym_mm: float = 0.0,
    canthal_tilt_deg: float = 4.0,
    pupil_ratio: float = 0.5,
) -> FaceMeasurements:
    return FaceMeasurements(
        mm_per_unit=0.35,
        pd_binocular_mm=pd_mm,
        pd_monocular_mm=PerSide(right=pd_mm / 2, left=pd_mm / 2),
        bridge=BridgeWidths(
            at_crest_mm=mid_bridge_mm - 5.0,
            below_10mm_mm=mid_bridge_mm,
            below_15mm_mm=mid_bridge_mm + 4.0,
        ),
        bridge_crest_height_mm=crest_mm,
        zygoma_width_mm=zygoma_mm,
        temple_width_mm=temple_width_mm,
        face_length_mm=zygoma_mm * EMPIRICAL_LENGTH_RATIO,
        jaw_width_mm=zygoma_mm * EMPIRICAL_JAW_RATIO,
        face_wrap_radius_mm=wrap_radius_mm,
        cheek_clearance_mm=PerSide(right=8.0, left=8.0),
        hinge_to_ear_mm=PerSide(right=hinge_right_mm, left=hinge_left_mm),
        ear_height_asymmetry_mm=ear_asym_mm,
        behind_ear={
            "right": BehindEar(drop_mm=14.0, angle_deg=45.0),
            "left": BehindEar(drop_mm=14.0, angle_deg=45.0),
        },
        vertex_estimate_mm=13.5,
        canthal_tilt_deg=PerSide(right=canthal_tilt_deg, left=canthal_tilt_deg),
        pupil_height_ratio=PerSide(right=pupil_ratio, left=pupil_ratio),
    )


AVERAGE: dict[str, float] = {}
NARROW: dict[str, float] = {
    "pd_mm": 58.0,
    "zygoma_mm": 116.0,
    "temple_width_mm": 124.0,
    "crest_mm": 4.5,
    "mid_bridge_mm": 15.0,
    "wrap_radius_mm": 90.0,
    "hinge_right_mm": 92.0,
    "hinge_left_mm": 93.0,
}
WIDE: dict[str, float] = {
    "pd_mm": 68.0,
    "zygoma_mm": 146.0,
    "temple_width_mm": 150.0,
    "crest_mm": 10.0,
    "mid_bridge_mm": 19.0,
    "wrap_radius_mm": 102.0,
    "hinge_right_mm": 106.0,
    "hinge_left_mm": 107.0,
}
FACES = [AVERAGE, NARROW, WIDE]
FACE_IDS = ["average", "narrow", "wide"]

RX_CASES = [
    None,
    Rx(od=RxEye(sphere=5.5), os=RxEye(sphere=5.0)),  # high plus
    Rx(od=RxEye(sphere=-7.0), os=RxEye(sphere=-6.5)),  # high minus
]
RX_IDS = ["no_rx", "high_plus", "high_minus"]


@pytest.mark.parametrize("rx", RX_CASES, ids=RX_IDS)
@pytest.mark.parametrize("face", FACES, ids=FACE_IDS)
def test_as_worn_always_within_spec_bands(params, face, rx):
    rec = recommend(make_measurements(**face), rx, None, params)
    assert 5.0 <= rec.as_worn.pantoscopic_deg <= 8.0
    assert 5.0 <= rec.as_worn.face_form_deg <= 10.0
    assert 12.0 <= rec.as_worn.vertex_mm <= 14.0


def test_a_monotonically_increases_with_zygoma(params):
    widths = [116.0, 124.0, 130.0, 138.0, 146.0]
    a_values = [
        recommend(make_measurements(zygoma_mm=w), None, None, params).frame.a_mm for w in widths
    ]
    assert all(a < b for a, b in pairwise(a_values)), a_values


@pytest.mark.parametrize("face", FACES, ids=FACE_IDS)
def test_temple_length_is_a_standard_size(params, face):
    rec = recommend(make_measurements(**face), None, None, params)
    assert rec.frame.temple_length_mm in params.temples.standard_lengths_mm
    trace = rec.rule_trace["frame.temple_length_mm"]
    if not trace.clamped:  # rounded UP: chosen standard covers the raw estimate
        assert rec.frame.temple_length_mm >= trace.raw_value


def test_ed_exceeds_a(params):
    rec = recommend(make_measurements(), None, None, params)
    assert rec.frame.ed_mm >= rec.frame.a_mm + params.frame.ed_margin_mm


def _numeric_paths(prefix: str, obj: dict) -> list[str]:
    paths: list[str] = []
    for key, val in obj.items():
        path = f"{prefix}.{key}"
        if isinstance(val, dict):
            paths.extend(_numeric_paths(path, val))
        elif isinstance(val, int | float) and not isinstance(val, bool):
            paths.append(path)
    return paths


def test_every_numeric_output_field_has_a_rule_trace_entry(params):
    rec = recommend(
        make_measurements(),
        Rx(od=RxEye(sphere=-1.5), os=RxEye(sphere=-1.25)),
        "progressive",
        params,
    )
    dump = rec.model_dump()
    paths: list[str] = []
    for section in ("frame", "as_worn", "optics", "nose_pads", "temples", "comfort"):
        paths.extend(_numeric_paths(section, dump[section]))
    assert len(paths) >= 20
    for path in paths:
        assert path in rec.rule_trace, f"missing rule_trace entry for {path}"
    for entry in rec.rule_trace.values():
        assert entry.rule_id


def test_clamped_flag_set_when_raw_falls_outside_band(params):
    # Extreme canthal tilt pushes the raw pantoscopic value above the 8 deg band edge.
    rec = recommend(make_measurements(canthal_tilt_deg=25.0), None, None, params)
    trace = rec.rule_trace["as_worn.pantoscopic_deg"]
    assert trace.raw_value > 8.0
    assert trace.clamped is True
    assert rec.as_worn.pantoscopic_deg == 8.0
    # And an in-band raw value is not flagged.
    calm = recommend(make_measurements(), None, None, params)
    assert calm.rule_trace["as_worn.pantoscopic_deg"].clamped is False


def test_high_minus_rx_pulls_vertex_toward_12(params):
    m = make_measurements()
    base = recommend(m, None, None, params)
    high_minus = recommend(m, Rx(od=RxEye(sphere=-7.0), os=RxEye(sphere=-6.5)), None, params)
    assert high_minus.as_worn.vertex_mm == pytest.approx(12.0)
    assert high_minus.as_worn.vertex_mm < base.as_worn.vertex_mm


def test_progressive_intent_raises_b_and_adds_near_inset(params):
    m = make_measurements()
    sv = recommend(m, None, "single_vision_distance", params)
    prog = recommend(m, None, "progressive", params)
    assert prog.frame.b_mm > sv.frame.b_mm
    assert prog.optics.inset_mm.right > sv.optics.inset_mm.right
    assert prog.optics.inset_mm.left > sv.optics.inset_mm.left
    assert prog.optics.oc_height_mm.right > sv.optics.oc_height_mm.right


def test_ear_asymmetry_raises_temple_on_higher_ear_side(params):
    # Positive asymmetry: subject's RIGHT ear sits HIGHER -> raise the right temple
    # (raising a temple lowers that side of the frame front, leveling it).
    rec = recommend(make_measurements(ear_asym_mm=2.0), None, None, params)
    assert rec.temples.raise_mm.right == pytest.approx(2.0)
    assert rec.temples.raise_mm.left == 0.0
    assert any("Raise right temple 2.0 mm" in n and "sits higher" in n for n in rec.notes)
    # Negative asymmetry: LEFT ear higher -> raise the left temple.
    rec_l = recommend(make_measurements(ear_asym_mm=-1.5), None, None, params)
    assert rec_l.temples.raise_mm.left == pytest.approx(1.5)
    assert rec_l.temples.raise_mm.right == 0.0
    assert any("Raise left temple 1.5 mm" in n for n in rec_l.notes)


def test_low_crest_gets_adjustable_pad_note_and_smaller_dbl(params):
    low = recommend(make_measurements(**NARROW), None, None, params)
    assert any("Low bridge crest" in n and "adjustable pads" in n for n in low.notes)
    high = recommend(make_measurements(**{**NARROW, "crest_mm": 10.0}), None, None, params)
    assert low.frame.dbl_mm < high.frame.dbl_mm  # adjustable-pad strategy shrinks DBL


@pytest.mark.parametrize("face", FACES, ids=FACE_IDS)
def test_notes_nonempty_with_concrete_numbers(params, face):
    rec = recommend(make_measurements(**face), None, None, params)
    assert rec.notes
    assert any(NUMBER_WITH_UNIT.search(n) for n in rec.notes)


def test_identity_and_mono_pd_override(params):
    m = make_measurements()
    rec = recommend(m, None, None, params)
    assert rec.ruleset_version == params.ruleset_version == "1.0.0"
    assert len(rec.recommendation_id) == 32
    assert rec.recommendation_id != recommend(m, None, None, params).recommendation_id
    override = PerSide(right=30.0, left=33.0)
    rec_o = recommend(m, None, None, params, pd_monocular=override)
    assert rec_o.optics.pd_monocular_mm.right == 30.0
    assert rec_o.optics.pd_monocular_mm.left == 33.0
    # Wider mono PD on the left -> smaller inward inset on the left.
    assert rec_o.optics.inset_mm.left < rec_o.optics.inset_mm.right


def test_comfort_proxies_bounded_and_track_pressure(params):
    for face in FACES:
        rec = recommend(make_measurements(**face), None, None, params)
        for value in (
            rec.comfort.predicted_slip,
            rec.comfort.nose_pressure,
            rec.comfort.temple_pressure,
        ):
            assert 0.0 <= value <= 1.0
    narrow_head = recommend(make_measurements(temple_width_mm=120.0), None, None, params)
    wide_head = recommend(make_measurements(temple_width_mm=145.0), None, None, params)
    assert wide_head.comfort.temple_pressure > narrow_head.comfort.temple_pressure
