"""Download the MediaPipe FaceLandmarker model bundle (stdlib only).

Usage: uv run python scripts/download_models.py [--force]
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request

# Version-pinned (/1/) for reproducibility; "latest" also exists upstream.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main() -> None:
    from glassfit.config import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    dest = get_settings().landmarker_model_path
    if dest.exists() and not args.force:
        print(f"already present: {dest} ({dest.stat().st_size:,} bytes) — use --force to refresh")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {MODEL_URL}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp:  # noqa: S310
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"download failed ({exc}) — check your network and retry")
    dest.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"wrote {dest} ({len(data):,} bytes)\nsha256 {digest}")


if __name__ == "__main__":
    main()
