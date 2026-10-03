"""
rubric.py — Reference triage rubric used to label the benchmark cases.

This is the answer key. It is intentionally independent of the production
engine: it does not import knowledge_graph/ and was written before the
graph-based scoring. It encodes wound-care principles the original rule set
does not model:

  * Ischemia + tissue loss is limb-threatening (critical limb ischemia), so
    even modest necrosis is escalated when Peripheral Artery Disease is present.
  * Systemic risk factors compound: a stalled wound in a patient with several
    healing impairments is more urgent than one with a single impairment.
  * A wound that is not shrinking warrants at least monitoring.

Severity: CRITICAL > HIGH > MEDIUM > LOW > OK. Highest matching tier wins.
"""

from __future__ import annotations

PRIORITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "OK"]


def systemic_burden(patient: dict) -> int:
    """Count independent systemic impairments to wound healing."""
    comorbidities = patient.get("comorbidities", [])
    return sum([
        patient.get("blood_glucose", 0.0) > 180,
        patient.get("serum_albumin", 4.0) < 3.0,
        patient.get("mobility_score", 10) < 4,
        "Peripheral Artery Disease" in comorbidities,
        "Type 2 Diabetes" in comorbidities,
        "Obesity" in comorbidities,
    ])


def reference_priority(area_delta: float, tissue: dict, patient: dict) -> str:
    """Return the expected priority for one case."""
    black  = tissue.get("black", 0.0)
    yellow = tissue.get("yellow", 0.0)
    red    = tissue.get("red", 0.0)

    ischemic    = "Peripheral Artery Disease" in patient.get("comorbidities", [])
    hyperglycemic = patient.get("blood_glucose", 0.0) > 180
    not_healing = area_delta >= 0
    worsening   = area_delta > 0.5
    burden      = systemic_burden(patient)

    if black > 15 or (black > 5 and ischemic) or (worsening and burden >= 4):
        return "CRITICAL"

    if not_healing and (yellow > 10 or hyperglycemic or ischemic or burden >= 3):
        return "HIGH"
    if black > 5 and burden >= 2:
        return "HIGH"

    if (
        patient.get("serum_albumin", 4.0) < 3.0
        or patient.get("mobility_score", 10) < 4
        or burden >= 3
        or (not_healing and burden >= 2)
    ):
        return "MEDIUM"

    if (patient.get("post_op_day", 0) > 14 and red < 60) or not_healing or burden >= 2:
        return "LOW"

    return "OK"
