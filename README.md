# GlassFit

Local, private eyewear fitting: scan your face in the browser, enter your pupillary distance
(PD), and get optician-grade facial measurements plus frame-size and fitting recommendations.

> **Disclaimer:** GlassFit is a fitting aid, not a medical device. It does not measure or
> produce prescriptions — bring its output to an optician, don't replace one with it.

## How it works

1. The browser captures ~6 camera frames and POSTs them to a local Python backend.
2. MediaPipe **FaceLandmarker** detects 478 3-D face landmarks (468 mesh + 10 iris points).
3. Your typed PD converts landmark units to millimeters (`mm_per_unit = pd_mm / iris-center distance`).
4. A measurement extractor computes optician measurements (bridge widths, zygoma/temple width,
   canthal tilt, vertex estimate, …).
5. A versioned rules engine turns measurements into a recommendation: frame A/B/DBL/ED/temple,
   pantoscopic tilt (5–8°), face-form/wrap, vertex, nose-pad setup, temple bends, comfort proxies.
6. Your fit feedback (pressure scores, slippage, adjustments) is logged to SQLite — the training
   set for a future ML residual model that learns corrections to the rules.

## Privacy

Everything runs and stays on your machine:

- The server binds `127.0.0.1` only. No telemetry, no cloud calls.
- Raw camera frames are processed **in memory and never written to disk** by default
  (debug opt-in: `GLASSFIT_SAVE_FRAMES=1`).
- Only derived landmark coordinates + measurements are persisted, in `data/runtime/glassfit.db`
  (gitignored). Deleting that one file erases all biometric data.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (pins and auto-installs Python 3.12 — MediaPipe does
not support newer interpreters).

```bash
uv sync --extra scan                        # install incl. mediapipe
uv run python scripts/download_models.py    # fetch face_landmarker.task (~3 MB, one time)
uv run glassfit                             # serve on http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>, allow camera access, scan, enter your PD, read your results.
Interactive API docs: <http://127.0.0.1:8000/docs>.

## Project layout

```
backend/glassfit/    FastAPI app: api/ schemas/ vision/ measure/ rules/ catalog/ services/ storage/
backend/tests/       pytest suite (runs WITHOUT mediapipe via committed landmark fixtures)
frontend/            static HTML/CSS/JS wizard (no build step), served by the backend
data/seed/           committed frame catalog seed
data/models|runtime/ gitignored: downloaded models, SQLite DB
training/            dataset export (implemented) + ML residual-model scaffold (Phase 3)
scripts/             utilities (model download)
```

## API overview

All JSON under `/api/v1` (multipart for scan):

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/scan` | camera frames → canonical 478-pt landmarks + overlay geometry + quality report |
| `POST /api/v1/measurements` | landmarks + pd_mm → named mm measurements |
| `POST /api/v1/recommendations` | landmarks/measurements + pd_mm (+ rx, lens intent) → full fit recommendation |
| `GET  /api/v1/frames` | sample frame catalog (Phase 2 adds ranked matching) |
| `POST /api/v1/feedback` | log fit feedback (training data) |
| `GET  /api/v1/health` | liveness + versions |

## Development

```bash
uv sync                      # light install — no mediapipe needed for the test suite
uv run pytest                # unit tests run on committed fixture landmarks
uv run ruff check .
uv run ruff format .
uv run pytest -m mediapipe -o addopts=""   # real-model integration tests (needs --extra scan + model)
```

## Data & training

Every scan/recommendation/feedback writes joined SQLite rows (with `engine_version` provenance).
Export the accumulated training table:

```bash
uv sync --extra train
uv run python -m glassfit_training.dataset export
```

## Roadmap

- **Phase 1 (this):** scan → measurements → recommendation → feedback logging, end to end.
- **Phase 2:** frame-catalog matching — ranked shortlist of real frames vs your recommendation.
- **Phase 3:** ML residual model (scikit-learn HistGradientBoosting) learning corrections to the
  rules engine from logged feedback; per-user personalization.
