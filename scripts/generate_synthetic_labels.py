"""Generate SYNTHETIC training labels from stored scans/recommendations.

Real human labels take months of real usage to accumulate. To prove the learning
loop end-to-end (features -> training -> evaluation -> artifact), this script
simulates a rater over every stored recommendation:

- A HIDDEN preference function scores each candidate frame. It deliberately
  differs from the matcher's fit_score: different component weights PLUS a taste
  term the components do not encode at all (a preference for panto/round shapes
  and light frames). If the trained model beats the ``1 + 4*fit_score`` baseline,
  it has learned exactly that hidden signal — which is the point of the test.
- Comfort feedback rows are simulated from the measurements with noise.

All rows are marked ``user_id='synthetic-v1'`` and are distinguishable from real
labels forever; delete them with:
    DELETE FROM match_ratings WHERE user_id='synthetic-v1';
    DELETE FROM feedback WHERE user_id='synthetic-v1';

Usage: uv run python scripts/generate_synthetic_labels.py [--db PATH] [--seed N]
"""

import argparse
import random
from pathlib import Path

from glassfit.catalog.match import (
    DEFAULT_PARAMS,
    context_for_recommendation,
    frame_front_width_mm,
    match_frames,
)
from glassfit.config import get_settings
from glassfit.rules.params import load_rule_params
from glassfit.schemas import AdjustmentsMade, FeedbackIn, MatchRatingIn
from glassfit.storage.db import connect
from glassfit.storage.repo import Repo
from glassfit_training.evaluate import heuristic_rating

SYNTH_USER = "synthetic-v1"

# Hidden preference: NOT the matcher's weights (bridge + shape dominate here).
PREF_WEIGHTS = {
    "optical_centration": 0.15,
    "bridge_fit": 0.30,
    "width_fit": 0.10,
    "lens_height": 0.10,
    "temple_fit": 0.00,
    "wrap_harmony": 0.05,
    "weight_slip": 0.05,
    "shape_affinity": 0.25,
}
TASTE_SHAPES = {"panto", "round"}  # invisible to the components — must be learned
TASTE_BONUS = 0.12
LIGHT_FRAME_G = 16.0
LIGHT_BONUS = 0.06


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def synth_rating(match, rng: random.Random) -> int:
    pref = sum(PREF_WEIGHTS[k] * v for k, v in match.components.items())
    if match.frame.shape in TASTE_SHAPES:
        pref += TASTE_BONUS
    if match.frame.weight_g <= LIGHT_FRAME_G:
        pref += LIGHT_BONUS
    pref = _clip(pref + rng.gauss(0.0, 0.07), 0.0, 1.0)
    return round(heuristic_rating(pref))  # pref is 0..1, so this lands in 1..5


def synth_feedback(rec_row, match, rng: random.Random) -> FeedbackIn:
    m = rec_row["measurements"]
    crest = m.bridge_crest_height_mm
    nose = _clip(0.75 - crest / 25.0 + match.frame.weight_g / 90.0 + rng.gauss(0, 0.08), 0, 1)
    front = frame_front_width_mm(
        match.frame.a_mm, match.frame.dbl_mm, DEFAULT_PARAMS.endpiece_total_mm
    )
    temple = _clip(0.25 + max(0.0, m.temple_width_mm - front) / 25.0 + rng.gauss(0, 0.08), 0, 1)
    slips = rng.random() < _clip(0.15 + (6.0 - crest) / 18.0 + match.frame.weight_g / 120.0, 0, 0.9)
    adjustments = None
    if rng.random() < 0.35:
        adjustments = AdjustmentsMade(panto_delta_deg=round(rng.gauss(0.0, 1.2), 1))
    return FeedbackIn(
        recommendation_id=rec_row["id"],
        frame_id=match.frame.frame_id,
        nose_pressure=round(heuristic_rating(nose)),
        temple_pressure=round(heuristic_rating(temple)),
        slips=slips,
        cheek_touch=rng.random() < 0.12,
        adjustments=adjustments,
        user_id=SYNTH_USER,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=get_settings().db_path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames-per-rec", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    conn = connect(args.db)
    repo = Repo(conn)
    catalog = repo.list_frames(limit=1000)
    low_crest = load_rule_params(get_settings().rules_path).nose_pads.low_crest_adjustable_mm

    rec_ids = [r["id"] for r in conn.execute("SELECT id FROM recommendations").fetchall()]
    ratings = feedback = 0
    for rec_id in rec_ids:
        row = repo.get_recommendation(rec_id)
        ctx = context_for_recommendation(row["recommendation"], row["measurements"], low_crest)
        matches = match_frames(catalog, ctx, limit=args.frames_per_rec)
        for match in matches:
            repo.save_match_rating(
                MatchRatingIn(
                    recommendation_id=rec_id,
                    frame_id=match.frame.frame_id,
                    rating=synth_rating(match, rng),
                    fit_score=match.fit_score,
                    components=match.components,
                    user_id=SYNTH_USER,
                )
            )
            ratings += 1
        if matches:
            repo.save_feedback(synth_feedback(row, matches[0], rng))
            feedback += 1
    conn.close()
    print(
        f"{len(rec_ids)} recommendations -> {ratings} synthetic ratings, {feedback} feedback rows"
    )


if __name__ == "__main__":
    main()
