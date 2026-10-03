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
python -m evals.triage.run_eval --label <name>  # defaults to the test split
```

## Protocol

1. The rubric and cases were committed **before** the graph-based scoring was written.
2. Engine thresholds may be tuned on the `dev` split only.
3. Headline numbers are reported on the `test` split.

## Limitations

The cases are synthetic and the rubric encodes published wound-care principles
(ischemic tissue loss, compounding systemic risk) as written by the developer,
not by a clinician. Numbers measure agreement with this rubric, not clinical
validity. A clinician review of a sample of `cases.json` labels is the next step.
