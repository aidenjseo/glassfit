"""Real-image evaluation harness: feed face photos through the LIVE GlassFit stack.

For each image the full HTTP pipeline runs — POST /api/v1/scan (the image repeated as a
frontal burst) then POST /api/v1/recommendations — and the outputs are checked against
anthropometric plausibility bands (PD is assumed 63 mm, the adult average, so absolute
values carry that assumption; the point is crash-freedom + plausible geometry at scale).

Usage:
    uv run glassfit                                   # in another terminal
    uv run python scripts/eval_real_faces.py IMG [IMG ...]
    uv run python scripts/eval_real_faces.py --quiet DIR_GLOB...   # batch: failures + summary
    uv run python scripts/eval_real_faces.py --expect-reject ROTATED_IMG
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

BASE = "http://127.0.0.1:8000/api/v1"
ASSUMED_PD_MM = 63.0

# (label, extractor, low, high) — bands for an adult face scaled at PD 63
BANDS = [
    ("zygoma_mm", lambda m: m["zygoma_width_mm"], 95.0, 170.0),
    ("temple_mm", lambda m: m["temple_width_mm"], 85.0, 155.0),
    # face_length uses landmark 10 (near-hairline) -> physiognomic-ish height; the jaw
    # proxy (58/288) sits wider than true gonion. Bands calibrated on 70+ real portraits.
    ("face_length_mm", lambda m: m["face_length_mm"], 130.0, 200.0),
    ("jaw_mm", lambda m: m["jaw_width_mm"], 95.0, 160.0),
    ("bridge_crest_w_mm", lambda m: m["bridge"]["at_crest_mm"], 6.0, 32.0),
    ("mono_pd_r_mm", lambda m: m["pd_monocular_mm"]["right"], 22.0, 41.0),
    ("mono_pd_l_mm", lambda m: m["pd_monocular_mm"]["left"], 22.0, 41.0),
    ("canthal_r_deg", lambda m: m["canthal_tilt_deg"]["right"], -18.0, 18.0),
    ("hinge_ear_r_mm", lambda m: m["hinge_to_ear_mm"]["right"], 30.0, 140.0),
]
FRAME_BANDS = [
    ("A_mm", lambda r: r["frame"]["a_mm"], 40.0, 62.0),
    ("DBL_mm", lambda r: r["frame"]["dbl_mm"], 12.0, 26.0),
    ("temple_len_mm", lambda r: r["frame"]["temple_length_mm"], 130.0, 155.0),
    ("panto_deg", lambda r: r["as_worn"]["pantoscopic_deg"], 5.0, 8.0),
    ("wrap_deg", lambda r: r["as_worn"]["face_form_deg"], 5.0, 10.0),
    ("vertex_mm", lambda r: r["as_worn"]["vertex_mm"], 12.0, 14.0),
]
# measurement keys collected for the batch summary distributions
SUMMARY_KEYS = ("zygoma_mm", "temple_mm", "face_length_mm", "jaw_mm", "mono_pd_r_mm")


def _multipart(fields: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, filename, data in fields:
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _post(path: str, data: bytes, content_type: str) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def evaluate(path: Path, expect_reject: bool, quiet: bool) -> dict:
    """Run one image through scan -> recommendation; returns a result record."""
    started = time.perf_counter()
    blob = path.read_bytes()
    body, ctype = _multipart([("frames", f"f{i}.jpg", blob) for i in range(3)])
    status, scan = _post("/scan", body, ctype)

    if expect_reject:
        rejected = status == 422
        code = scan.get("error", {}).get("code", "?") if rejected else "ACCEPTED"
        print(f"{path.name:34s} expect-reject: {'PASS' if rejected else 'FAIL'} ({code})")
        return {"name": path.name, "ok": rejected, "reason": None}

    if status != 200:
        err = scan.get("error", {})
        reason = err.get("code", f"HTTP{status}")
        detail = ""
        reports = err.get("details", {}).get("frame_reports")
        if reports:
            detail = f" ({reports[0].get('reject_reason')})"
        print(f"{path.name:34s} SCAN REJECTED {reason}{detail}")
        return {"name": path.name, "ok": False, "reason": f"{reason}{detail}"}

    payload = json.dumps({"pd_mm": ASSUMED_PD_MM, "scan_id": scan["scan_id"]}).encode()
    status, rec = _post("/recommendations", payload, "application/json")
    if status != 200:
        print(f"{path.name:34s} RECOMMENDATION FAILED {status}: {rec}")
        return {"name": path.name, "ok": False, "reason": f"rec HTTP{status}"}

    m = rec["measurements"]
    values: dict[str, float] = {}
    failures: list[str] = []
    for source, bands in ((m, BANDS), (rec, FRAME_BANDS)):
        for label, get, low, high in bands:
            value = get(source)
            if source is m:
                values[label] = value
            if not low <= value <= high:
                failures.append(f"{label}={value:.1f}!in[{low},{high}]")
    latency = time.perf_counter() - started
    result = {
        "name": path.name,
        "ok": not failures,
        "reason": "; ".join(failures) or None,
        "values": values,
        "suspect": m["quality"]["scale_suspect"],
        "latency_s": latency,
    }
    if failures or not quiet:
        report = scan["quality"]["frame_reports"][0]
        verdict = "PASS" if not failures else "FAIL " + result["reason"]
        pose = f"{report['yaw_deg']:+.0f}/{report['pitch_deg']:+.0f}/{report['roll_deg']:+.0f}"
        print(
            f"{path.name:34s} {verdict}  "
            f"[pose {pose}°  zygoma {values['zygoma_mm']:.0f}  "
            f"A/DBL/tpl {rec['frame']['a_mm']:.0f}/"
            f"{rec['frame']['dbl_mm']:.0f}/{rec['frame']['temple_length_mm']:.0f}]"
        )
    return result


def summarize(results: list[dict]) -> None:
    scanned = [r for r in results if "values" in r]
    rejected = [r for r in results if "values" not in r]
    passed = [r for r in results if r["ok"]]
    print("\n================ BATCH SUMMARY ================")
    print(f"images: {len(results)}  scanned: {len(scanned)}  band-pass: {len(passed)}")
    if rejected:
        reasons = Counter(r["reason"] or "?" for r in rejected)
        print("scan rejections:")
        for reason, count in reasons.most_common():
            print(f"  {count:3d} × {reason}")
    if scanned:
        print(f"scale_suspect flags: {sum(r['suspect'] for r in scanned)}")
        lat = [r["latency_s"] for r in scanned]
        print(f"latency per face: mean {statistics.mean(lat):.2f}s  max {max(lat):.2f}s")
        print("measurement distributions (PD assumed 63mm):")
        for key in SUMMARY_KEYS:
            vals = [r["values"][key] for r in scanned]
            print(
                f"  {key:16s} mean {statistics.mean(vals):6.1f}  "
                f"sd {statistics.stdev(vals) if len(vals) > 1 else 0.0:5.1f}  "
                f"range [{min(vals):.1f}, {max(vals):.1f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--quiet", action="store_true", help="print failures + summary only")
    parser.add_argument(
        "--expect-reject",
        action="store_true",
        help="assert the images are REJECTED (rotated/profile shots)",
    )
    args = parser.parse_args()
    results = [evaluate(p, args.expect_reject, args.quiet) for p in args.images]
    if not args.expect_reject and len(results) > 1:
        summarize(results)
    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
