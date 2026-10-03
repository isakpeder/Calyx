"""
generate_cases.py — Build the frozen triage benchmark (cases.json).

Samples synthetic patient + scan cases from clinically plausible
distributions, labels each with the reference rubric, and assigns a
dev/test split. Tune only on dev; report only on test.

Usage:
    python -m evals.triage.generate_cases
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .rubric import reference_priority

SEED = 2026
N_CASES = 500
N_DEV = 150
OUT_PATH = Path(__file__).parent / "cases.json"

COMORBIDITY_RATES = {
    "Type 2 Diabetes":           0.45,
    "Hypertension":              0.45,
    "Obesity":                   0.35,
    "Peripheral Artery Disease": 0.20,
    "Malnutrition":              0.15,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sample_tissue(rng: random.Random) -> dict:
    roll = rng.random()
    if roll < 0.60:
        black = rng.uniform(0, 5)
    elif roll < 0.85:
        black = rng.uniform(5, 15)
    else:
        black = rng.uniform(15, 40)
    yellow = rng.uniform(0, 100 - black) * rng.uniform(0.2, 0.8)
    red = 100 - black - yellow
    return {"red": round(red, 1), "yellow": round(yellow, 1), "black": round(black, 1)}


def _sample_case(rng: random.Random, idx: int) -> dict:
    comorbidities = [c for c, p in COMORBIDITY_RATES.items() if rng.random() < p]
    diabetic = "Type 2 Diabetes" in comorbidities
    malnourished = "Malnutrition" in comorbidities

    patient = {
        "name": f"Case {idx:03d}",
        "comorbidities": comorbidities,
        "blood_glucose": round(_clip(rng.gauss(185, 45) if diabetic else rng.gauss(110, 20), 70, 400), 1),
        "serum_albumin": round(_clip(rng.gauss(2.6, 0.3) if malnourished else rng.gauss(3.6, 0.45), 1.5, 5.0), 2),
        "mobility_score": rng.choices(range(11), weights=[2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3])[0],
        "post_op_day": rng.randint(1, 40),
    }
    area_delta = round(rng.gauss(-0.4, 0.8), 2)
    tissue = _sample_tissue(rng)

    return {
        "case_id": f"T{idx:03d}",
        "patient": patient,
        "area_delta": area_delta,
        "tissue_ratios": tissue,
        "expected_priority": reference_priority(area_delta, tissue, patient),
    }


def main() -> None:
    rng = random.Random(SEED)
    cases = [_sample_case(rng, i) for i in range(N_CASES)]

    order = list(range(N_CASES))
    rng.shuffle(order)
    dev_ids = set(order[:N_DEV])
    for i, case in enumerate(cases):
        case["split"] = "dev" if i in dev_ids else "test"

    OUT_PATH.write_text(json.dumps(cases, indent=1) + "\n")

    for split in ("dev", "test"):
        counts = Counter(c["expected_priority"] for c in cases if c["split"] == split)
        print(f"{split}: {sum(counts.values())} cases  {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
