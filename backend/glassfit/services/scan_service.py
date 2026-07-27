"""Scan orchestration: uploaded frames -> canonical landmark set + overlay geometry.

Privacy: raw frame bytes are processed in memory and NEVER written to disk unless
``settings.save_frames`` is explicitly enabled (debug opt-in).
"""

from __future__ import annotations

import threading
import uuid

import numpy as np

from glassfit import __version__
from glassfit.config import Settings
from glassfit.errors import InvalidImage, MultipleFaces, NoFaceDetected, PoorScanQuality
from glassfit.measure import indices as idx
from glassfit.measure.scale import resolve_sides
from glassfit.schemas import (
    FrameReport,
    ImageSize,
    LandmarkSet,
    OverlaySegment,
    Point2,
    ScanQuality,
    ScanResponse,
)
from glassfit.storage.repo import Repo
from glassfit.vision.aggregate import aggregate_landmarks
from glassfit.vision.base import DetectionResult, LandmarkBackend
from glassfit.vision.decode import decode_image
from glassfit.vision.quality import evaluate_frame

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


def run_scan(
    frame_blobs: list[bytes],
    *,
    detector: LandmarkBackend,
    repo: Repo,
    settings: Settings,
    lock: threading.Lock | None = None,
) -> ScanResponse:
    if not frame_blobs:
        raise InvalidImage("no frames uploaded")
    if len(frame_blobs) > MAX_FRAMES:
        raise InvalidImage(f"too many frames ({len(frame_blobs)}), max {MAX_FRAMES}")
    lock = lock or threading.Lock()

    reports: list[FrameReport] = []
    accepted_sets: list[np.ndarray] = []
    accepted_reports: list[FrameReport] = []
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
            accepted_sets.append(np.asarray(det.landmarks, dtype=float))
            accepted_reports.append(report)
            image_w, image_h = det.image_width, det.image_height

    reject_reasons = [r.reject_reason for r in reports if not r.accepted]
    if not accepted_sets:
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
    if len(accepted_sets) < settings.min_accepted_frames:
        raise PoorScanQuality(
            f"Only {len(accepted_sets)} usable frame(s); "
            f"need {settings.min_accepted_frames} — hold still and face the camera",
            details=_report_details(reports),
        )

    canonical, dispersion_px = aggregate_landmarks(accepted_sets, image_w, image_h)
    warnings: list[str] = []
    ok = dispersion_px <= settings.max_landmark_dispersion_px
    if not ok:
        warnings.append(
            f"landmark_dispersion_{dispersion_px:.1f}px_above_"
            f"{settings.max_landmark_dispersion_px:.1f}px — consider rescanning"
        )
    quality = ScanQuality(
        frames_received=len(frame_blobs),
        frames_used=len(accepted_sets),
        frame_reports=reports,
        landmark_dispersion_px=float(dispersion_px),
        ok=ok,
        warnings=warnings,
    )

    scan_id = uuid.uuid4().hex
    landmark_set = LandmarkSet(
        points=[(float(p[0]), float(p[1]), float(p[2])) for p in canonical],
        image_width=image_w,
        image_height=image_h,
    )

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

    best = min(accepted_reports, key=_pose_error)
    repo.save_scan(scan_id, landmark_set, quality, settings.landmark_model_name, __version__)
    _maybe_save_frames(frame_blobs, scan_id, settings)

    return ScanResponse(
        scan_id=scan_id,
        landmarks=landmark_set,
        landmarks_px=[(float(p[0] * image_w), float(p[1] * image_h)) for p in canonical],
        key_points=key_points,
        overlay_segments=overlay_segments,
        image_size=ImageSize(width=image_w, height=image_h),
        best_frame_index=best.index,
        quality=quality,
    )
