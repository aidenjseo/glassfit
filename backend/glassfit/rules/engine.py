"""Pure rules engine: FaceMeasurements (+ optional Rx / lens intent) -> Recommendation.

Every numeric output field gets a RuleTrace entry keyed by its output field path
(e.g. "as_worn.pantoscopic_deg") so Phase-3 residual models can learn per-rule corrections.
All traced outputs flow through one ``emit`` helper (clamp -> trace -> round), so a trace
key can never drift from its output field.

Units: lengths mm, angles deg. Sides are SUBJECT-anatomical (right = subject's right eye = OD).

Sign convention (matches measure/extractor.py and schemas/measurements.py):
`ear_height_asymmetry_mm > 0` means the SUBJECT'S RIGHT ear sits HIGHER than the left.
Raising a temple (bending it up at the endpiece) LOWERS that side of the frame front, so
the temple on the HIGHER-ear side is raised to level the frame.
"""

import math
from uuid import uuid4

from ..catalog.match import frame_front_width_mm
from ..schemas import (
    AsWorn,
    Comfort,
    FaceMeasurements,
    FrameDims,
    LensIntent,
    NosePads,
    Optics,
    PerSide,
    Recommendation,
    RuleTrace,
    Rx,
    Temples,
)
from .params import RuleParams

_SIDES = ("right", "left")


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    clamped = min(max(value, lo), hi)
    return clamped, clamped != value


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _round_step(value: float, step: float) -> float:
    """Round to the nearest multiple of `step`, halves away from zero-ward drift (half-up)."""
    return round(math.floor(value / step + 0.5) * step, 6)


def _round_up_standard(value: float, options: list[float]) -> tuple[float, bool]:
    """Smallest standard size >= value; (max, clamped=True) when value exceeds every option."""
    for opt in sorted(options):
        if opt >= value - 1e-9:
            return opt, False
    return max(options), True


def recommend(
    m: FaceMeasurements,
    rx: Rx | None,
    lens_intent: LensIntent | None,
    params: RuleParams,
    pd_monocular: PerSide | None = None,
) -> Recommendation:
    """Map facial measurements to a full frame/fit recommendation. Pure — no I/O."""
    p = params
    rnd = p.rounding
    g = p.rx_guardrails
    trace: dict[str, RuleTrace] = {}
    notes: list[str] = []

    def emit(
        field: str,
        rule_id: str,
        inputs: dict[str, float],
        raw: float,
        lo: float = -math.inf,
        hi: float = math.inf,
        *,
        step: float | None = None,
        decimals: int | None = None,
        clamped: bool | None = None,
    ) -> float:
        """Clamp -> trace (with the PRE-clamp raw) -> round; returns the final value."""
        val, was_clamped = _clamp(raw, lo, hi)
        trace[field] = RuleTrace(
            rule_id=rule_id,
            inputs={k: float(v) for k, v in inputs.items()},
            raw_value=float(raw),
            clamped=was_clamped if clamped is None else clamped,
        )
        if step is not None:
            return _round_step(val, step)
        if decimals is not None:
            return round(val, decimals)
        return val

    intent: str = lens_intent or "single_vision_distance"
    sphere = {"right": rx.od.sphere if rx else 0.0, "left": rx.os.sphere if rx else 0.0}
    max_plus = max(sphere["right"], sphere["left"], 0.0)
    most_minus = min(sphere["right"], sphere["left"], 0.0)
    max_abs_sphere = max(abs(sphere["right"]), abs(sphere["left"]))

    # --- frame: A from zygoma minus margin ---------------------------------
    az = p.frame.a_from_zygoma
    a_raw = az.slope * m.zygoma_width_mm + az.intercept_mm - az.margin_mm
    a_mm = emit(
        "frame.a_mm",
        "a_from_zygoma_v1",
        {"zygoma_width_mm": m.zygoma_width_mm, "slope": az.slope, "margin_mm": az.margin_mm},
        a_raw,
        az.min_mm,
        az.max_mm,
        step=rnd.length_step_mm,
    )

    # --- frame: DBL from mid-bridge width + pad strategy -------------------
    crest = m.bridge_crest_height_mm
    adjustable_pads = crest < p.nose_pads.low_crest_adjustable_mm
    mid_bridge = m.bridge.below_10mm_mm
    dblp = p.frame.dbl
    offset = dblp.offset_adjustable_pads_mm if adjustable_pads else dblp.offset_fixed_pads_mm
    dbl_mm = emit(
        "frame.dbl_mm",
        "dbl_from_mid_bridge_v1",
        {
            "mid_bridge_width_mm": mid_bridge,
            "offset_mm": offset,
            "adjustable_pads": float(adjustable_pads),
        },
        mid_bridge + offset,
        dblp.min_mm,
        dblp.max_mm,
        step=rnd.length_step_mm,
    )

    # --- frame PD cross-check vs patient PD --------------------------------
    frame_pd = a_mm + dbl_mm
    pd = m.pd_binocular_mm
    dec_per_lens = (frame_pd - pd) / 2.0
    notes.append(
        f"Frame PD {frame_pd:.1f} mm vs pupillary distance {pd:.1f} mm "
        f"({dec_per_lens:+.1f} mm decentration per lens)."
    )
    if abs(dec_per_lens) > p.frame.frame_pd.decentration_warn_mm:
        direction = "narrower" if dec_per_lens > 0 else "wider"
        notes.append(
            f"Decentration {abs(dec_per_lens):.1f} mm per lens exceeds "
            f"{p.frame.frame_pd.decentration_warn_mm:.1f} mm — a {direction} frame would "
            f"reduce edge thickness and prism risk."
        )

    # --- frame: B from pupil height ratio + lens intent --------------------
    bp = p.frame.b_from_pupil
    mean_ratio = (m.pupil_height_ratio.right + m.pupil_height_ratio.left) / 2.0
    b_raw = a_mm * bp.base_b_over_a + bp.pupil_ratio_gain_mm * (mean_ratio - 0.5)
    min_b = bp.min_b_mm
    if intent == "progressive":
        b_raw += bp.progressive_extra_b_mm
        min_b = max(min_b, bp.min_b_progressive_mm)
    b_mm = emit(
        "frame.b_mm",
        "b_from_pupil_v1",
        {
            "a_mm": a_mm,
            "mean_pupil_height_ratio": mean_ratio,
            "progressive": float(intent == "progressive"),
        },
        b_raw,
        min_b,
        bp.max_b_mm,
        step=rnd.length_step_mm,
    )

    # --- frame: ED >= A + margin -------------------------------------------
    ed_mm = emit(
        "frame.ed_mm",
        "ed_from_a_v1",
        {"a_mm": a_mm, "ed_margin_mm": p.frame.ed_margin_mm},
        a_mm + p.frame.ed_margin_mm,
        step=rnd.length_step_mm,
    )

    # --- frame: temple length rounded UP to a standard size ----------------
    tp = p.temples
    hinge_max = max(m.hinge_to_ear_mm.right, m.hinge_to_ear_mm.left)
    temple_raw = hinge_max + tp.bend_allowance_mm
    temple_mm, temple_clamped = _round_up_standard(temple_raw, tp.standard_lengths_mm)
    emit(
        "frame.temple_length_mm",
        "temple_standard_size_v1",
        {"hinge_to_ear_max_mm": hinge_max, "bend_allowance_mm": tp.bend_allowance_mm},
        temple_raw,
        clamped=temple_clamped,
    )
    if temple_clamped:
        notes.append(
            f"Estimated temple need {temple_raw:.0f} mm exceeds the longest standard size "
            f"({temple_mm:.0f} mm) — expect a shallower bend."
        )

    # --- as-worn: pantoscopic from canthal tilt + high-plus guardrail ------
    ap = p.as_worn.pantoscopic
    mean_canthal = (m.canthal_tilt_deg.right + m.canthal_tilt_deg.left) / 2.0
    panto_raw = ap.default_deg + ap.canthal_tilt_gain * mean_canthal
    high_plus = max_plus >= g.high_plus_sphere_d
    if high_plus:
        panto_raw -= g.high_plus_panto_reduction_deg
        notes.append(
            f"High-plus Rx ({max_plus:+.2f} D) — pantoscopic tilt reduced by "
            f"{g.high_plus_panto_reduction_deg:.1f} deg."
        )
    panto = emit(
        "as_worn.pantoscopic_deg",
        "panto_from_canthal_v1",
        {"mean_canthal_tilt_deg": mean_canthal, "high_plus": float(high_plus)},
        panto_raw,
        ap.min_deg,
        ap.max_deg,
        step=rnd.angle_step_deg,
    )

    # --- as-worn: face form from wrap radius + strong-Rx cap ---------------
    fp = p.as_worn.face_form
    ff_raw = fp.base_deg + fp.gain_deg_per_mm * (fp.wrap_radius_ref_mm - m.face_wrap_radius_mm)
    strong_rx = max_abs_sphere >= g.strong_rx_abs_sphere_d
    ff_hi = min(fp.max_deg, g.strong_rx_wrap_cap_deg) if strong_rx else fp.max_deg
    face_form = emit(
        "as_worn.face_form_deg",
        "face_form_from_wrap_v1",
        {"face_wrap_radius_mm": m.face_wrap_radius_mm, "strong_rx": float(strong_rx)},
        ff_raw,
        fp.min_deg,
        ff_hi,
        step=rnd.angle_step_deg,
    )
    if strong_rx and ff_raw > ff_hi:
        notes.append(
            f"Strong Rx (|sphere| {max_abs_sphere:.2f} D) — face-form wrap capped at "
            f"{ff_hi:.1f} deg to limit oblique astigmatism."
        )

    # --- as-worn: vertex, pulled to the minimum for high minus -------------
    vp = p.as_worn.vertex
    vertex_raw = vp.default_mm + vp.measured_gain * (m.vertex_estimate_mm - vp.default_mm)
    high_minus = most_minus <= g.high_minus_sphere_d
    if high_minus:
        vertex_raw = g.high_minus_vertex_target_mm
    vertex = emit(
        "as_worn.vertex_mm",
        "vertex_default_rx_v1",
        {"vertex_estimate_mm": m.vertex_estimate_mm, "high_minus": float(high_minus)},
        vertex_raw,
        vp.min_mm,
        vp.max_mm,
        step=rnd.length_step_mm,
    )
    if high_minus:
        notes.append(
            f"High-minus Rx ({most_minus:+.2f} D) — fit vertex close at {vertex:.1f} mm to "
            f"preserve effective power."
        )
    notes.append(
        f"As worn: pantoscopic {panto:.1f} deg, face form {face_form:.1f} deg, "
        f"vertex {vertex:.1f} mm."
    )

    # --- optics: monocular PDs (user-provided wins over measured) ----------
    mono = pd_monocular if pd_monocular is not None else m.pd_monocular_mm
    mono_rule = "mono_pd_provided_v1" if pd_monocular is not None else "mono_pd_measured_v1"
    for side in _SIDES:
        value = getattr(mono, side)
        emit(f"optics.pd_monocular_mm.{side}", mono_rule, {"pd_monocular_mm": value}, value)

    # --- optics: OC heights from B x target pupil fraction -----------------
    frac = bp.pupil_target_frac.get(intent, bp.pupil_target_frac["default"])
    oc: dict[str, float] = {}
    for side in _SIDES:
        ratio = getattr(m.pupil_height_ratio, side)
        oc[side] = emit(
            f"optics.oc_height_mm.{side}",
            "oc_height_v1",
            {"b_mm": b_mm, "target_frac": frac, "pupil_height_ratio": ratio},
            b_mm * frac + p.optics.oc_ratio_gain_mm * (ratio - mean_ratio),
            0.0,
            b_mm,
            step=rnd.length_step_mm,
        )
    if intent in ("progressive", "office"):
        notes.append(
            f"{intent.capitalize()} fitting heights: {oc['right']:.1f} mm (OD) / "
            f"{oc['left']:.1f} mm (OS) from the lens bottom."
        )

    # --- optics: inset = distance decentration + near inset ----------------
    near_base = p.optics.near_inset_mm.get(intent, p.optics.near_inset_mm["default"])
    inset: dict[str, float] = {}
    for side in _SIDES:
        mono_side = getattr(mono, side)
        dist_dec = frame_pd / 2.0 - mono_side
        near_clamped = False
        near_val = 0.0
        if near_base > 0:
            near_raw = near_base + p.optics.inset_per_diopter_mm * sphere[side]
            near_val, near_clamped = _clamp(near_raw, 0.0, p.optics.near_inset_max_mm)
        inset[side] = emit(
            f"optics.inset_mm.{side}",
            "inset_split_by_mono_pd_v1",
            {
                "frame_pd_mm": frame_pd,
                "pd_monocular_mm": mono_side,
                "sphere_d": sphere[side],
                "near_inset_base_mm": near_base,
            },
            dist_dec + near_val,
            step=rnd.inset_step_mm,
            clamped=near_clamped,
        )

    # --- nose pads ---------------------------------------------------------
    npd = p.nose_pads
    st = npd.size_thresholds
    if crest < st.low_crest_mm:
        size = "L"
    elif crest > st.high_crest_mm:
        size = "S"
    else:
        size = "M"
    emit(
        "nose_pads.size",
        "pad_size_from_crest_v1",
        {"bridge_crest_height_mm": crest, "low_mm": st.low_crest_mm, "high_mm": st.high_crest_mm},
        crest,
    )
    if adjustable_pads:
        notes.append(
            f"Low bridge crest ({crest:.1f} mm) — prefer adjustable pads or a low-bridge fit."
        )

    sp = npd.splay
    splay = emit(
        "nose_pads.splay_deg",
        "pad_splay_from_bridge_v1",
        {"mid_bridge_width_mm": mid_bridge},
        sp.default_deg + sp.gain_deg_per_mm * (mid_bridge - sp.bridge_width_ref_mm),
        sp.min_deg,
        sp.max_deg,
        step=rnd.angle_step_deg,
    )

    flare = emit(
        "nose_pads.flare_deg",
        "pad_flare_default_v1",
        {},
        npd.flare_default_deg,
        step=rnd.angle_step_deg,
    )

    dp = npd.drop
    drop = emit(
        "nose_pads.drop_mm",
        "pad_drop_from_crest_v1",
        {"bridge_crest_height_mm": crest},
        dp.default_mm + dp.gain_mm_per_mm * (dp.crest_ref_mm - crest),
        dp.min_mm,
        dp.max_mm,
        step=rnd.length_step_mm,
    )

    # --- temples: bend point, tip angle, per-side raise --------------------
    bend: dict[str, float] = {}
    for side in _SIDES:
        hinge_side = getattr(m.hinge_to_ear_mm, side)
        bend[side] = emit(
            f"temples.bend_point_mm_from_hinge.{side}",
            "temple_bend_point_v1",
            {"hinge_to_ear_mm": hinge_side, "bend_point_offset_mm": tp.bend_point_offset_mm},
            hinge_side + tp.bend_point_offset_mm,
            step=rnd.length_step_mm,
        )

    angles = [be.angle_deg for be in m.behind_ear.values()] or [tp.tip_angle_default_deg]
    tip_raw = sum(angles) / len(angles)
    tip = emit(
        "temples.tip_angle_deg",
        "temple_tip_angle_v1",
        {"mean_behind_ear_angle_deg": tip_raw},
        tip_raw,
        tp.tip_angle_min_deg,
        tp.tip_angle_max_deg,
        step=rnd.angle_step_deg,
    )

    asym = m.ear_height_asymmetry_mm
    raw_raise = {"right": 0.0, "left": 0.0}
    if abs(asym) >= tp.raise_threshold_mm:
        # asym > 0 = right ear HIGHER; raising that temple lowers its frame side.
        higher_side = "right" if asym > 0 else "left"
        raw_raise[higher_side] = abs(asym)
    raise_mm: dict[str, float] = {}
    for side in _SIDES:
        raise_mm[side] = emit(
            f"temples.raise_mm.{side}",
            "temple_raise_from_ear_asym_v1",
            {"ear_height_asymmetry_mm": asym, "raise_threshold_mm": tp.raise_threshold_mm},
            raw_raise[side],
            0.0,
            tp.raise_max_mm,
            step=rnd.length_step_mm,
        )
        if raise_mm[side] > 0:
            notes.append(
                f"Raise {side} temple {raise_mm[side]:.1f} mm to level the frame "
                f"({side} ear sits higher)."
            )

    # --- comfort proxies ---------------------------------------------------
    cp = p.comfort
    front_width = frame_front_width_mm(a_mm, dbl_mm, 2.0 * cp.endpiece_mm)
    overhang = m.temple_width_mm - front_width
    tpp = cp.temple_pressure
    temple_pressure = emit(
        "comfort.temple_pressure",
        "comfort_temple_pressure_v1",
        {"temple_width_mm": m.temple_width_mm, "frame_front_mm": front_width},
        (overhang - tpp.zero_at_mm) / (tpp.one_at_mm - tpp.zero_at_mm),
        0.0,
        1.0,
        decimals=rnd.comfort_decimals,
    )
    if overhang >= cp.spring_hinge_note_mm:
        notes.append(
            f"Head width {m.temple_width_mm:.1f} mm exceeds the frame front "
            f"(frame PD {frame_pd:.1f} mm + end pieces = {front_width:.1f} mm) by "
            f"{overhang:.1f} mm — spring hinges recommended."
        )

    npp = cp.nose_pressure
    nose_pressure = emit(
        "comfort.nose_pressure",
        "comfort_nose_pressure_v1",
        {"bridge_crest_height_mm": crest},
        npp.base + (npp.crest_ref_mm - crest) / npp.crest_range_mm,
        0.0,
        1.0,
        decimals=rnd.comfort_decimals,
    )

    sl = cp.slip
    lens_area = a_mm * b_mm
    crest_term = _clamp01((sl.crest_ref_mm - crest) / sl.crest_range_mm)
    area_term = _clamp01((lens_area - sl.lens_area_ref_mm2) / sl.lens_area_range_mm2)
    predicted_slip = emit(
        "comfort.predicted_slip",
        "comfort_slip_v1",
        {"bridge_crest_height_mm": crest, "lens_area_mm2": lens_area},
        sl.crest_weight * crest_term + sl.lens_area_weight * area_term,
        0.0,
        1.0,
        decimals=rnd.comfort_decimals,
    )

    return Recommendation(
        recommendation_id=uuid4().hex,
        ruleset_version=p.ruleset_version,
        frame=FrameDims(
            a_mm=a_mm, b_mm=b_mm, dbl_mm=dbl_mm, ed_mm=ed_mm, temple_length_mm=temple_mm
        ),
        as_worn=AsWorn(pantoscopic_deg=panto, face_form_deg=face_form, vertex_mm=vertex),
        optics=Optics(
            pd_monocular_mm=PerSide(right=mono.right, left=mono.left),
            oc_height_mm=PerSide(right=oc["right"], left=oc["left"]),
            inset_mm=PerSide(right=inset["right"], left=inset["left"]),
        ),
        nose_pads=NosePads(size=size, splay_deg=splay, flare_deg=flare, drop_mm=drop),
        temples=Temples(
            bend_point_mm_from_hinge=PerSide(right=bend["right"], left=bend["left"]),
            tip_angle_deg=tip,
            raise_mm=PerSide(right=raise_mm["right"], left=raise_mm["left"]),
        ),
        comfort=Comfort(
            predicted_slip=predicted_slip,
            nose_pressure=nose_pressure,
            temple_pressure=temple_pressure,
        ),
        notes=notes,
        rule_trace=trace,
    )
