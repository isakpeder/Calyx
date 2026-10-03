"""
run_calibration.py — Wound-area accuracy with vs without coin calibration.

For every synthetic scene, measures area three ways:
  coin      full pipeline (analyze_frame) on the photo with a coin
  fallback  full pipeline on the same photo without a coin (fixed 0.026 cm/px)
  oracle    wound mask + true scale; isolates segmentation error from calibration

Metrics per condition
---------------------
median_error_pct   median |measured - true| / true
within_10pct       share of scans measured within 10% of true area
accuracy_pct       100 - mean absolute % error (floored at 0)

Usage:
    python -m evals.vision.run_calibration [--n 200] [--label baseline]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from vision import analyze_frame, compute_wound_area, detect_wound_mask

from .scenes import make_scene

RESULTS_DIR = Path(__file__).parents[1] / "results"
SEED_OFFSET = 10_000


def _summarize(errors: list[float]) -> dict:
    return {
        "n": len(errors),
        "median_error_pct": round(statistics.median(errors), 1),
        "within_10pct": round(sum(e <= 10 for e in errors) / len(errors), 3),
        "accuracy_pct": round(max(0.0, 100 - statistics.mean(errors)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--label", default="current", help="name for the results file")
    args = parser.parse_args()

    errors: dict[str, list[float]] = defaultdict(list)
    by_tone: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coin_found = coin_correct = 0
    radius_errors: list[float] = []
    rows = []

    for i in range(args.n):
        scene = make_scene(SEED_OFFSET + i)
        truth = scene.true_area_cm2

        with_coin = analyze_frame(scene.image)
        without = analyze_frame(scene.image_no_coin)
        oracle_area, _ = compute_wound_area(detect_wound_mask(scene.image), 1 / scene.px_per_cm)

        measured = {"coin": with_coin["area_cm2"], "fallback": without["area_cm2"], "oracle": oracle_area}
        for cond, area in measured.items():
            err = abs(area - truth) / truth * 100
            errors[cond].append(err)
            by_tone[scene.skin_tone][cond].append(err)

        if with_coin["coin_found"]:
            coin_found += 1
            detected_r = (2.426 / 2) / with_coin["scale_cm_per_px"]
            radius_errors.append(abs(detected_r - scene.coin_radius_px) / scene.coin_radius_px * 100)
            if radius_errors[-1] <= 10:
                coin_correct += 1

        rows.append({"seed": SEED_OFFSET + i, "tone": scene.skin_tone, "px_per_cm": round(scene.px_per_cm, 1),
                     "true_cm2": round(truth, 2), **{k: round(v, 2) for k, v in measured.items()},
                     "coin_found": with_coin["coin_found"]})

    results = {
        "n_scenes": args.n,
        "conditions": {cond: _summarize(errs) for cond, errs in errors.items()},
        "by_skin_tone": {tone: {cond: _summarize(e) for cond, e in conds.items()}
                         for tone, conds in sorted(by_tone.items())},
        "coin_detection_rate": round(coin_found / args.n, 3),
        "coin_correct_rate": round(coin_correct / args.n, 3),
        "coin_radius_median_error_pct": round(statistics.median(radius_errors), 1) if radius_errors else None,
        "scenes": rows,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"vision_calibration_{args.label}.json"
    out.write_text(json.dumps(results, indent=1) + "\n")

    print(f"n={args.n}  coin found {results['coin_detection_rate']:.1%}  "
          f"correct coin {results['coin_correct_rate']:.1%}  "
          f"radius err {results['coin_radius_median_error_pct']}%")
    print(f"{'':10}{'median err':>11}{'within 10%':>12}{'accuracy':>10}")
    for cond, s in results["conditions"].items():
        print(f"{cond:10}{s['median_error_pct']:>10}%{s['within_10pct']:>12.1%}{s['accuracy_pct']:>9}%")
    print("by skin tone (median err / within 10%):")
    for tone, conds in results["by_skin_tone"].items():
        cells = "  ".join(f"{c} {conds[c]['median_error_pct']:>6}% / {conds[c]['within_10pct']:>6.1%}" for c in conds)
        print(f"  {tone:8} n={conds['coin']['n']:3}  {cells}")


if __name__ == "__main__":
    main()
