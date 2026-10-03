"""
run_eval.py — Score the triage engine against the frozen benchmark.

Metrics
-------
accuracy           exact priority match over all cases
critical_recall    share of truly CRITICAL cases the engine labels CRITICAL
macro_f1           F1 averaged over the five priority levels
ranking_accuracy   over every pair of cases with different expected severity,
                   share the dashboard sort orders correctly (ties count 0.5)

Usage:
    python -m evals.triage.run_eval [--split test|dev|all]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from knowledge_graph import evaluate_healing

from .rubric import PRIORITIES

CASES_PATH = Path(__file__).parent / "cases.json"
RESULTS_DIR = Path(__file__).parents[1] / "results"


def _rank(priority: str) -> int:
    return PRIORITIES.index(priority)


def _run_system(cases: list[dict]) -> list[dict]:
    """Run the engine on every case; return (priority, risk_score) per case."""
    outputs = []
    for case in cases:
        result = evaluate_healing(case["area_delta"], case["tissue_ratios"], case["patient"])
        outputs.append({"priority": result["priority"], "risk_score": result.get("risk_score", 0.0)})
    return outputs


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _ranking_accuracy(cases: list[dict], outputs: list[dict]) -> float:
    """Pairwise agreement between the dashboard order and the expected severity order."""
    keys = [(_rank(o["priority"]), -o["risk_score"]) for o in outputs]
    truth = [_rank(c["expected_priority"]) for c in cases]
    score = pairs = 0.0
    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            if truth[i] == truth[j]:
                continue
            pairs += 1
            if keys[i] == keys[j]:
                score += 0.5
            elif (keys[i] < keys[j]) == (truth[i] < truth[j]):
                score += 1
    return score / pairs


def score(cases: list[dict], outputs: list[dict]) -> dict:
    expected = [c["expected_priority"] for c in cases]
    predicted = [o["priority"] for o in outputs]

    per_class = {}
    for p in PRIORITIES:
        tp = sum(e == p and q == p for e, q in zip(expected, predicted))
        fp = sum(e != p and q == p for e, q in zip(expected, predicted))
        fn = sum(e == p and q != p for e, q in zip(expected, predicted))
        per_class[p] = {"support": tp + fn, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                        "f1": round(_f1(tp, fp, fn), 3)}

    confusion = Counter(f"{e}->{q}" for e, q in zip(expected, predicted) if e != q)

    return {
        "n_cases": len(cases),
        "accuracy": round(sum(e == q for e, q in zip(expected, predicted)) / len(cases), 3),
        "critical_recall": per_class["CRITICAL"]["recall"],
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(PRIORITIES), 3),
        "ranking_accuracy": round(_ranking_accuracy(cases, outputs), 3),
        "per_class": per_class,
        "top_errors": dict(confusion.most_common(8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--label", default="current", help="name for the results file")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text())
    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]

    metrics = score(cases, _run_system(cases))
    metrics["split"] = args.split

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"triage_{args.label}_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"[{args.label} / {args.split}] n={metrics['n_cases']}")
    for key in ("accuracy", "critical_recall", "macro_f1", "ranking_accuracy"):
        print(f"  {key:<17} {metrics[key]:.3f}")
    print(f"  top errors        {metrics['top_errors']}")
    print(f"  -> {out.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
