"""Scan orchestration: uploaded frames -> canonical landmark set + overlay geometry.

Privacy: raw frame bytes are processed in memory and NEVER written to disk unless
``settings.save_frames`` is explicitly enabled (debug opt-in).
"""

from __future__ import annotations

import logging
import threading
import uuid

import numpy as np

from glassfit import __version__
from glassfit.config import Settings
from glassfit.errors import InvalidImage, MultipleFaces, NoFaceDetected, PoorScanQuality
from glassfit.measure import indices as idx
from glassfit.measure.scale import resolve_sides
from glassfit.measure.side_views import summarize_side_views
from glassfit.schemas import (
    FrameReport,
    ImageSize,
    LandmarkSet,
    OverlaySegment,
    Point2,
    ProbeResponse,
    ScanQuality,
    ScanResponse,
    SideViewSummary,
)
from glassfit.storage.repo import Repo
from glassfit.vision.aggregate import aggregate_landmarks
from glassfit.vision.base import DetectionResult, LandmarkBackend
from glassfit.vision.decode import decode_image
from glassfit.vision.quality import evaluate_frame, pose_of

logger = logging.getLogger("glassfit.scan")

MAX_FRAMES = 15


def _report_details(reports: list[FrameReport]) -> dict:
    return {"frame_reports": [r.model_dump() for r in reports]}


def _pose_error(report: FrameReport) -> float:
    return abs(report.yaw_deg or 0.0) + abs(report.pitch_deg or 0.0)


def _maybe_save_frames(blobs: list[bytes], scan_id: str, settings: Settings) -> None:
    if not settings.save_frames:
        return
    settings.save_frames_dir.mkdir(parents=True, exist_ok=True)
    for i, blob in enumerate(blobs):
        (settings.save_frames_dir / f"{scan_id}_{i:02d}.jpg").write_bytes(blob)


def _process_side_frames(
    side_blobs: list[bytes],
    *,
    detector: LandmarkBackend,
    settings: Settings,
    lock: threading.Lock,
) -> tuple[SideViewSummary, list[str]]:
    """Head-turn frames -> per-subject-side hinge-to-ear refinement.

    Non-blocking by design: unusable side frames only produce warnings — the scan
    still succeeds on the frontal burst alone.
    """
    usable: list[LandmarkSet] = []
    for blob in side_blobs:
        try:
            img = decode_image(blob)
        except InvalidImage:
            continue
        with lock:
            det = detector.detect(img)
        if det.face_count != 1 or det.landmarks.size == 0:
            continue
        yaw = abs(pose_of(det)[0])
        if not settings.min_side_yaw_deg <= yaw <= settings.max_side_yaw_deg:
            continue
        usable.append(LandmarkSet.from_array(det.landmarks, det.image_width, det.image_height))
    counts, medians = summarize_side_views(usable)
    warnings: list[str] = []
    # Turning the head to the subject's LEFT exposes the subject's RIGHT side.
    if counts["right"] == 0:
        warnings.append("side_view_right_not_captured — left head-turn frames unusable")
    if counts["left"] == 0:
        warnings.append("side_view_left_not_captured — right head-turn frames unusable")
    summary = SideViewSummary(
        right_frames=counts["right"], left_frames=counts["left"], hinge_to_ear_mm=medians
    )
    return summary, warnings


def run_probe(
    frame_blob: bytes,
    *,
    detector: LandmarkBackend,
    settings: Settings,
    lock: threading.Lock | None = None,
) -> ProbeResponse:
    """One live-preview frame -> distance/framing coaching. Nothing is persisted."""
    lock = lock or threading.Lock()
    img = decode_image(frame_blob)
    with lock:
        det: DetectionResult = detector.detect(img)
    if det.face_count == 0 or det.landmarks.size == 0:
        return ProbeResponse(
            face_count=det.face_count,
            guidance="no_face",
            message="Center your face in the oval",
        )
    xs = det.landmarks[:, 0]
    frac = float(xs.max() - xs.min())
    yaw, pitch, roll = pose_of(det)
    if frac < settings.probe_ideal_min_frac:
        guidance, message = "closer", "Move a little closer"
    elif frac > settings.probe_ideal_max_frac:
        guidance, message = "back", "Back up a bit"
    else:
        guidance, message = "ok", "Distance good — hold there ✓"
    return ProbeResponse(
        face_count=det.face_count,
        face_width_frac=frac,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        guidance=guidance,
        message=message,
    )


def run_scan(
    frame_blobs: list[bytes],
    *,
    detector: LandmarkBackend,
    repo: Repo,
    settings: Settings,
    side_blobs: list[bytes] | None = None,
    lock: threading.Lock | None = None,
) -> ScanResponse:
    if not frame_blobs:
        raise InvalidImage("no frames uploaded")
    if len(frame_blobs) + len(side_blobs or []) > MAX_FRAMES:
        raise InvalidImage(f"too many frames, max {MAX_FRAMES} total")
    lock = lock or threading.Lock()

    reports: list[FrameReport] = []
    accepted: list[tuple[FrameReport, np.ndarray]] = []
    image_w = image_h = 0
    for i, blob in enumerate(frame_blobs):
        try:
            img = decode_image(blob)
        except InvalidImage as exc:
            raise InvalidImage(exc.message, details={"frame_index": i}) from exc
        with lock:
            det: DetectionResult = detector.detect(img)
        report = evaluate_frame(
            det,
            max_yaw_deg=settings.max_yaw_deg,
            max_pitch_deg=settings.max_pitch_deg,
            max_roll_deg=settings.max_roll_deg,
            min_face_width_frac=settings.min_face_width_frac,
            index=i,
        )
        reports.append(report)
        if report.accepted:
            accepted.append((report, np.asarray(det.landmarks, dtype=float)))
            image_w, image_h = det.image_width, det.image_height

    reject_reasons = [r.reject_reason for r in reports if not r.accepted]
    if len(accepted) < settings.min_accepted_frames:
        logger.info(
            "scan rejected: %d/%d frontal frames usable; reasons=%s",
            len(accepted),
            len(frame_blobs),
            reject_reasons,
        )
    if not accepted:
        if reject_reasons and all(r == "no_face" for r in reject_reasons):
            raise NoFaceDetected(
                "No face detected — face the camera straight on with even lighting",
                details=_report_details(reports),
            )
        if reject_reasons.count("multiple_faces") * 2 >= len(reports):
            raise MultipleFaces(
                "More than one face in frame — scan with a single face visible",
                details=_report_details(reports),
            )
        raise PoorScanQuality("No usable frames captured", details=_report_details(reports))
    if len(accepted) < settings.min_accepted_frames:
        raise PoorScanQuality(
            f"Only {len(accepted)} usable frame(s); "
            f"need {settings.min_accepted_frames} — hold still and face the camera",
            details=_report_details(reports),
        )

    # The aggregate is Procrustes-aligned to the first set, and the pixel outputs
    # (key_points / overlay_segments) are drawn over the BEST frame — so the best
    # (most frontal) frame leads as the alignment reference.
    best_i = min(range(len(accepted)), key=lambda i: _pose_error(accepted[i][0]))
    best = accepted[best_i][0]
    ordered_sets = [accepted[best_i][1], *(s for i, (_, s) in enumerate(accepted) if i != best_i)]
    canonical, dispersion_px = aggregate_landmarks(ordered_sets, image_w, image_h)
    warnings: list[str] = []
    ok = dispersion_px <= settings.max_landmark_dispersion_px
    if not ok:
        warnings.append(
            f"landmark_dispersion_{dispersion_px:.1f}px_above_"
            f"{settings.max_landmark_dispersion_px:.1f}px — consider rescanning"
        )
    side_summary: SideViewSummary | None = None
    if side_blobs:
        side_summary, side_warnings = _process_side_frames(
            side_blobs, detector=detector, settings=settings, lock=lock
        )
        warnings.extend(side_warnings)
    quality = ScanQuality(
        frames_received=len(frame_blobs) + len(side_blobs or []),
        frames_used=len(accepted),
        frame_reports=reports,
        landmark_dispersion_px=float(dispersion_px),
        ok=ok,
        warnings=warnings,
        side_views=side_summary,
    )

    scan_id = uuid.uuid4().hex
    landmark_set = LandmarkSet.from_array(canonical, image_w, image_h)
    key_points, overlay_segments = _overlay_geometry(canonical, image_w, image_h)

    repo.save_scan(scan_id, landmark_set, quality, settings.landmark_model_name, __version__)
    _maybe_save_frames(list(frame_blobs) + list(side_blobs or []), scan_id, settings)

    return ScanResponse(
        scan_id=scan_id,
        landmarks=landmark_set,
        key_points=key_points,
        overlay_segments=overlay_segments,
        image_size=ImageSize(width=image_w, height=image_h),
        best_frame_index=best.index,
        quality=quality,
    )


def _overlay_geometry(
    canonical: np.ndarray, image_w: int, image_h: int
) -> tuple[dict[str, Point2], list[OverlaySegment]]:
    """Named pixel anchors + labeled measurement segments for the review overlay."""

    def px(i: int) -> Point2:
        return (float(canonical[i, 0] * image_w), float(canonical[i, 1] * image_h))

    sides = resolve_sides(canonical)
    key_points: dict[str, Point2] = {
        "pupil_right": px(sides.right.iris_center),
        "pupil_left": px(sides.left.iris_center),
        "nasion": px(idx.NASION),
        "zygoma_right": px(sides.right.face_side),
        "zygoma_left": px(sides.left.face_side),
        "temple_right": px(sides.right.temple),
        "temple_left": px(sides.left.temple),
        "chin": px(idx.CHIN),
    }
    nasion = key_points["nasion"]
    overlay_segments = [
        OverlaySegment(
            name="pd", start=key_points["pupil_right"], end=key_points["pupil_left"], label="PD"
        ),
        OverlaySegment(
            name="zygoma_width",
            start=key_points["zygoma_right"],
            end=key_points["zygoma_left"],
            label="Cheekbone width",
        ),
        OverlaySegment(
            name="temple_width",
            start=key_points["temple_right"],
            end=key_points["temple_left"],
            label="Temple width",
        ),
        OverlaySegment(
            name="bridge_crest",
            start=(nasion[0] - 14.0, nasion[1]),
            end=(nasion[0] + 14.0, nasion[1]),
            label="Bridge",
        ),
    ]
    return key_points, overlay_segments
