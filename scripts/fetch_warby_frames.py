"""Fetch Warby Parker product photos as LOCAL, PERSONAL-USE try-on art.

The try-on works out of the box with parametric SVG frames rendered from the
catalog's own dimensions. If you want photographic frames, this script downloads
front-view product imagery from warbyparker.com into ``data/tryon/<frame_id>.png``
— a gitignored folder, so the images stay on your machine.

  ⚠ Copyright: Warby Parker's product photography belongs to Warby Parker.
    Keep these files local and personal; do not commit or redistribute them.
    (data/ is gitignored precisely so this cannot happen by accident.)

The mapping below pairs each GlassFit catalog frame with a Warby style of a
similar shape. Product pages sometimes move; failures are reported and skipped —
the SVG fallback keeps the feature working regardless. You can also drop ANY
transparent front-view PNG into data/tryon/ named <frame_id>.png yourself.

Usage: uv run python scripts/fetch_warby_frames.py [--force]
"""

import argparse
import re
import sys
import urllib.error
import urllib.request

from glassfit.config import get_settings

# GlassFit frame_id -> Warby Parker eyeglasses slug of a comparable silhouette.
WARBY_MAP = {
    "GF-001": "percey",  # rectangle acetate
    "GF-002": "baird",  # panto
    "GF-003": "felix",  # square
    "GF-004": "york",  # aviator metal
    "GF-005": "gemma",  # cat-eye
    "GF-006": "ames",  # browline
    "GF-008": "britten",  # round metal
    "GF-010": "hughes",  # bold square acetate
    "GF-014": "chamberlain",  # wayfarer-ish
    "GF-018": "simon",  # panto metal
}

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 GlassFit-personal/0.1"
OG_IMAGE = re.compile(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"')


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = get_settings().tryon_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for frame_id, slug in WARBY_MAP.items():
        dest = out_dir / f"{frame_id}.png"
        if dest.exists() and not args.force:
            ok += 1
            continue
        page_url = f"https://www.warbyparker.com/eyeglasses/{slug}"
        try:
            page = fetch(page_url).decode("utf-8", errors="ignore")
            match = OG_IMAGE.search(page)
            if not match:
                print(f"{frame_id}: no og:image on {page_url} — skipped")
                continue
            image = fetch(match.group(1).replace("&amp;", "&"))
            if len(image) < 10_000:
                print(f"{frame_id}: image too small — skipped")
                continue
            dest.write_bytes(image)
            ok += 1
            print(f"{frame_id}: saved {slug} ({len(image):,} bytes)")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"{frame_id}: fetch failed ({exc}) — the SVG fallback will be used")
    print(f"\n{ok}/{len(WARBY_MAP)} frames have local art in {out_dir}")
    if ok == 0:
        sys.exit(
            "nothing fetched — warbyparker.com may be blocking scripted access; "
            "you can save front-view PNGs manually into data/tryon/<frame_id>.png"
        )


if __name__ == "__main__":
    main()
