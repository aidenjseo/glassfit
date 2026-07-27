"""Real-image evaluation harness: feed face photos through the LIVE GlassFit stack.

For each image the full HTTP pipeline runs — POST /api/v1/scan (the image repeated as a
frontal burst) then POST /api/v1/recommendations — and the outputs are checked against
anthropometric plausibility bands (PD is assumed 63 mm, the adult average, so absolute
values carry that assumption; the point is crash-freedom + plausible geometry).

Usage:
    uv run glassfit                                   # in another terminal
    uv run python scripts/eval_real_faces.py IMG [IMG ...]
    uv run python scripts/eval_real_faces.py --expect-reject ROTATED_IMG
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000/api/v1"
ASSUMED_PD_MM = 63.0

# (label, extractor, low, high) — bands for an adult face scaled at PD 63
BANDS = [
    ("zygoma_mm", lambda m: m["zygoma_width_mm"], 95.0, 170.0),
    ("temple_mm", lambda m: m["temple_width_mm"], 85.0, 155.0),
    ("bridge_crest_w_mm", lambda m: m["bridge"]["at_crest_mm"], 6.0, 32.0),
    ("mono_pd_r_mm", lambda m: m["pd_monocular_mm"]["right"], 24.0, 39.0),
    ("mono_pd_l_mm", lambda m: m["pd_monocular_mm"]["left"], 24.0, 39.0),
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


def evaluate(path: Path, expect_reject: bool) -> bool:
    blob = path.read_bytes()
    body, ctype = _multipart([("frames", f"f{i}.jpg", blob) for i in range(3)])
    status, scan = _post("/scan", body, ctype)

    if expect_reject:
        rejected = status == 422
        code = scan.get("error", {}).get("code", "?") if rejected else "ACCEPTED"
        print(f"{path.name:34s} expect-reject: {'PASS' if rejected else 'FAIL'} ({code})")
        return rejected

    if status != 200:
        err = scan.get("error", {})
        print(f"{path.name:34s} SCAN FAILED {status} {err.get('code')}: {err.get('message')}")
        return False

    report = scan["quality"]["frame_reports"][0]
    payload = json.dumps({"pd_mm": ASSUMED_PD_MM, "scan_id": scan["scan_id"]}).encode()
    status, rec = _post("/recommendations", payload, "application/json")
    if status != 200:
        print(f"{path.name:34s} RECOMMENDATION FAILED {status}: {rec}")
        return False

    m = rec["measurements"]
    failures = [
        f"{label}={value:.1f}!in[{low},{high}]"
        for source, bands in ((m, BANDS), (rec, FRAME_BANDS))
        for label, get, low, high in bands
        if not low <= (value := get(source)) <= high
    ]
    flags = []
    if m["quality"]["scale_suspect"]:
        flags.append("scale_suspect")
    verdict = "PASS" if not failures else "FAIL " + "; ".join(failures)
    print(
        f"{path.name:34s} {verdict}  "
        f"[pose {report['yaw_deg']:+.0f}/{report['pitch_deg']:+.0f}/{report['roll_deg']:+.0f}°  "
        f"zygoma {m['zygoma_width_mm']:.0f}  temple {m['temple_width_mm']:.0f}  "
        f"A/DBL/tpl {rec['frame']['a_mm']:.0f}/{rec['frame']['dbl_mm']:.0f}/"
        f"{rec['frame']['temple_length_mm']:.0f}  "
        f"panto {rec['as_worn']['pantoscopic_deg']:.1f}°"
        f"{'  ' + ','.join(flags) if flags else ''}]"
    )
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--expect-reject",
        action="store_true",
        help="assert the images are REJECTED (rotated/profile shots)",
    )
    args = parser.parse_args()
    results = [evaluate(p, args.expect_reject) for p in args.images]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
