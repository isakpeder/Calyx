"""
tune.py — Grid-search the knowledge-graph escalation thresholds on the dev split.

Runs the rules-only engine once per case to get its priority and graph risk
score, then simulates every monotone threshold set (including
"never escalate", so tuning can never do worse than rules alone). Objective: dev accuracy,
ties broken by critical recall. Never reads the test split.

Usage:
    python -m evals.triage.tune
"""

from __future__ import annotations

import itertools
import json

from knowledge_graph import evaluate_healing

from .rubric import PRIORITIES
from .run_eval import CASES_PATH

NEVER = float("inf")
GRID = [round(0.25 * i, 2) for i in range(1, 33)] + [NEVER]  # 0.25 … 8.0, or never escalate


def _escalate(rule_rank: int, risk: float, thresholds: tuple[float, ...]) -> int:
    for rank, t in enumerate(thresholds):  # rank 0=CRITICAL … 3=LOW
        if risk >= t:
            return min(rule_rank, rank)
    return rule_rank


def main() -> None:
    cases = [c for c in json.loads(CASES_PATH.read_text()) if c["split"] == "dev"]
    truth = [PRIORITIES.index(c["expected_priority"]) for c in cases]

    base = []
    for c in cases:
        r = evaluate_healing(c["area_delta"], c["tissue_ratios"], c["patient"], use_graph=False)
        base.append((PRIORITIES.index(r["priority"]), r["risk_score"]))

    n_critical = truth.count(0)
    best = None
    for combo in itertools.combinations(sorted(GRID, reverse=True), 4):  # strictly descending
        preds = [_escalate(rr, risk, combo) for rr, risk in base]
        acc = sum(p == t for p, t in zip(preds, truth)) / len(cases)
        crit = sum(p == 0 and t == 0 for p, t in zip(preds, truth)) / n_critical
        key = (round(acc, 4), round(crit, 4))
        if best is None or key > best[0]:
            best = (key, combo)

    (acc, crit), combo = best
    print(f"dev accuracy {acc:.3f}  critical recall {crit:.3f}")
    print("GRAPH_PRIORITY_THRESHOLDS =", dict(zip(PRIORITIES[:4], combo)))
    print(f"rules-only dev accuracy {sum(rr == t for (rr, _), t in zip(base, truth)) / len(cases):.3f}")


if __name__ == "__main__":
    main()
