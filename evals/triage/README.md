# Triage benchmark

Measures how well the reasoning engine assigns a priority (CRITICAL → OK) and
orders patients on the doctor dashboard.

| File | Purpose |
|---|---|
| `rubric.py` | Reference rubric (the answer key). Independent of `knowledge_graph/`. |
| `generate_cases.py` | Samples 500 seeded synthetic cases, labels them with the rubric, splits 150 dev / 350 test. |
| `cases.json` | The frozen benchmark. Regenerate only if the rubric changes. |
| `run_eval.py` | Scores the engine; writes `evals/results/triage_<label>_<split>.json`. |

```bash
python -m evals.triage.generate_cases           # only when the rubric changes
python -m evals.triage.tune                      # pick thresholds on dev
python -m evals.triage.run_eval --rules-only     # baseline (original engine)
python -m evals.triage.run_eval                  # rules + knowledge graph
```

## Results

Held-out `test` split, 350 cases:

| Metric | Rules only | Rules + knowledge graph |
|---|---|---|
| Accuracy | 74.3% | **79.1%** |
| Ranking accuracy | 88.0% | **93.7%** |
| Macro F1 | 0.710 | **0.775** |
| Critical recall | 64.0% | 66.7% |

Dev split (used for tuning): 78.0% → 84.0% accuracy. The dev/test gap is
expected overfitting from threshold tuning.

Ranking accuracy is the share of patient pairs with different expected
priority that the dashboard sort (priority, then risk score) puts in the
right order.

The graph mostly fixes under-triage of HIGH (71% → 87% recall) and LOW
(34% → 56%) cases. It does little for CRITICAL: 20 of 75 critical cases
are still rated HIGH, mainly ischemic wounds with 5–15% necrosis.

## Protocol

1. The rubric and cases were committed **before** the graph-based scoring was written.
2. Engine thresholds may be tuned on the `dev` split only.
3. Headline numbers are reported on the `test` split.

## Limitations

The cases are synthetic and the rubric encodes published wound-care principles
(ischemic tissue loss, compounding systemic risk) as written by the developer,
not by a clinician. Numbers measure agreement with this rubric, not clinical
validity. A clinician review of a sample of `cases.json` labels is the next step.
