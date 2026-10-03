"""
tests/test_knowledge_graph.py — Clinical knowledge graph + reasoning engine tests.

Test categories
---------------
1. build_graph       — node taxonomy, edge weights, caching
2. get_risk_factors  — comorbidity / biomarker activation and BFS reachability
3. get_outcome_scores — wound-state nodes, compounding, no double counting
4. evaluate_healing  — one test per triage rule, severity resolution, output shape
5. graph escalation  — escalates on compounding risk, never de-escalates
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledge_graph import build_graph, evaluate_healing, get_risk_factors  # noqa: E402
from knowledge_graph.reasoning import PRIORITY_ORDER  # noqa: E402
from knowledge_graph.graph import (  # noqa: E402
    OUTCOME_NODES,
    compute_risk_score,
    get_outcome_scores,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _healthy_patient(**overrides) -> dict:
    """Patient with no comorbidities and every biomarker in the normal range."""
    patient = {
        "name": "Test Patient",
        "comorbidities": [],
        "blood_glucose": 100.0,
        "serum_albumin": 4.0,
        "mobility_score": 8,
        "post_op_day": 5,
    }
    patient.update(overrides)
    return patient


HEALTHY_TISSUE = {"red": 90.0, "yellow": 10.0, "black": 0.0}
HEALING_DELTA = -1.0


# ===========================================================================
# 1. build_graph
# ===========================================================================

class TestBuildGraph:
    def test_every_node_has_a_known_type(self):
        G = build_graph()
        types = {d["type"] for _, d in G.nodes(data=True)}
        assert types <= {"condition", "biomarker", "wound_state", "outcome"}

    def test_outcome_nodes_are_sinks(self):
        G = build_graph()
        for node in OUTCOME_NODES:
            assert G.out_degree(node) == 0

    def test_edge_weights_are_probabilities(self):
        G = build_graph()
        for _, _, w in G.edges(data="weight"):
            assert 0.0 < w <= 1.0

    def test_graph_is_cached(self):
        assert build_graph() is build_graph()


# ===========================================================================
# 2. get_risk_factors
# ===========================================================================

class TestGetRiskFactors:
    def test_healthy_patient_has_no_risk_factors(self):
        assert get_risk_factors(_healthy_patient()) == []

    def test_hyperglycemia_reaches_infection_risk(self):
        risks = get_risk_factors(_healthy_patient(blood_glucose=220.0))
        assert "Infection_Risk" in risks

    def test_pad_reaches_necrosis_risk_through_perfusion(self):
        # Peripheral_Artery_Disease → Poor_Perfusion → Necrosis_Risk (depth 2)
        risks = get_risk_factors(_healthy_patient(comorbidities=["Peripheral Artery Disease"]))
        assert "Necrosis_Risk" in risks

    def test_low_albumin_activates_delayed_healing(self):
        risks = get_risk_factors(_healthy_patient(serum_albumin=2.5))
        assert "Delayed_Healing" in risks

    def test_low_mobility_activates_stagnation(self):
        risks = get_risk_factors(_healthy_patient(mobility_score=2))
        assert "Wound_Stagnation" in risks

    def test_unknown_comorbidity_is_ignored(self):
        assert get_risk_factors(_healthy_patient(comorbidities=["Asthma"])) == []

    def test_only_outcome_nodes_are_returned(self):
        risks = get_risk_factors(_healthy_patient(
            comorbidities=["Type 2 Diabetes", "Obesity", "Peripheral Artery Disease"],
            blood_glucose=250.0,
            serum_albumin=2.2,
            mobility_score=1,
        ))
        assert risks and set(risks) <= set(OUTCOME_NODES)


# ===========================================================================
# 3. get_outcome_scores
# ===========================================================================

STALLED_NECROTIC = {"tissue_ratios": {"red": 70.0, "yellow": 20.0, "black": 10.0}, "area_delta": 0.3}


class TestGetOutcomeScores:
    def test_wound_state_activates_outcomes_for_healthy_patient(self):
        scores = get_outcome_scores(_healthy_patient(), STALLED_NECROTIC)
        assert {"Necrosis_Risk", "Infection_Risk", "Wound_Stagnation"} <= set(scores)

    def test_without_wound_context_scores_only_patient_factors(self):
        assert get_outcome_scores(_healthy_patient(), None) == {}

    def test_independent_risk_factors_compound(self):
        one = compute_risk_score(get_outcome_scores(_healthy_patient(blood_glucose=220.0)))
        two = compute_risk_score(get_outcome_scores(_healthy_patient(blood_glucose=220.0, serum_albumin=2.5)))
        assert two > one

    def test_shared_mechanism_is_not_double_counted(self):
        # Diabetes acts on Infection_Risk only through Hyperglycemia, so adding
        # the diabetes label to an already-hyperglycemic patient must not
        # raise Infection_Risk.
        hyper = get_outcome_scores(_healthy_patient(blood_glucose=220.0))
        both = get_outcome_scores(_healthy_patient(blood_glucose=220.0, comorbidities=["Type 2 Diabetes"]))
        assert both["Infection_Risk"] == pytest.approx(hyper["Infection_Risk"])

    def test_pad_amplifies_necrosis_risk(self):
        base = get_outcome_scores(_healthy_patient(), STALLED_NECROTIC)
        pad = get_outcome_scores(_healthy_patient(comorbidities=["Peripheral Artery Disease"]), STALLED_NECROTIC)
        assert pad["Necrosis_Risk"] > base["Necrosis_Risk"]

    def test_scores_are_independent_of_input_order(self):
        a = _healthy_patient(comorbidities=["Obesity", "Malnutrition", "Type 2 Diabetes"])
        b = _healthy_patient(comorbidities=["Type 2 Diabetes", "Malnutrition", "Obesity"])
        assert get_outcome_scores(a, STALLED_NECROTIC) == get_outcome_scores(b, STALLED_NECROTIC)


# ===========================================================================
# 4. evaluate_healing
# ===========================================================================

class TestEvaluateHealing:
    def test_healthy_patient_is_ok(self):
        result = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient())
        assert result["priority"] == "OK"

    def test_necrosis_is_critical(self):
        tissue = {"red": 60.0, "yellow": 20.0, "black": 20.0}
        result = evaluate_healing(HEALING_DELTA, tissue, _healthy_patient())
        assert result["priority"] == "CRITICAL"

    def test_stalled_wound_with_slough_is_high(self):
        tissue = {"red": 70.0, "yellow": 30.0, "black": 0.0}
        result = evaluate_healing(0.2, tissue, _healthy_patient())
        assert result["priority"] == "HIGH"

    def test_hyperglycemia_with_stalled_wound_is_high(self):
        tissue = {"red": 95.0, "yellow": 5.0, "black": 0.0}
        result = evaluate_healing(0.0, tissue, _healthy_patient(blood_glucose=220.0))
        assert result["priority"] == "HIGH"

    def test_low_albumin_is_medium(self):
        result = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient(serum_albumin=2.5))
        assert result["priority"] == "MEDIUM"

    def test_low_mobility_is_medium(self):
        result = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient(mobility_score=2))
        assert result["priority"] == "MEDIUM"

    def test_delayed_granulation_is_low(self):
        tissue = {"red": 50.0, "yellow": 50.0, "black": 0.0}
        result = evaluate_healing(HEALING_DELTA, tissue, _healthy_patient(post_op_day=20))
        assert result["priority"] == "LOW"

    def test_highest_severity_wins_and_all_alerts_kept(self):
        tissue = {"red": 40.0, "yellow": 30.0, "black": 30.0}
        result = evaluate_healing(0.5, tissue, _healthy_patient(serum_albumin=2.5))
        assert result["priority"] == "CRITICAL"
        assert len(result["alerts"]) >= 3

    def test_result_has_frontend_keys(self):
        result = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient())
        for key in ("priority", "alerts", "reasoning", "active_risk_factors", "recommended_action"):
            assert key in result

    @pytest.mark.parametrize("delta", [-2.0, 0.0, 2.0])
    def test_reasoning_mentions_patient_name(self, delta):
        result = evaluate_healing(delta, HEALTHY_TISSUE, _healthy_patient())
        assert "Test Patient" in result["reasoning"]


# ===========================================================================
# 5. graph escalation
# ===========================================================================

HIGH_RISK_PATIENT = _healthy_patient(
    comorbidities=["Type 2 Diabetes", "Obesity", "Peripheral Artery Disease"],
    blood_glucose=250.0,
    serum_albumin=2.4,
    mobility_score=2,
)


class TestGraphEscalation:
    def test_compounding_risk_escalates_above_rules(self):
        # Necrosis 10% is below the 15% CRITICAL rule, but with ischemia,
        # hyperglycemia and malnutrition on a stalled wound it is limb-threatening.
        args = (0.3, STALLED_NECROTIC["tissue_ratios"], HIGH_RISK_PATIENT)
        rules_only = evaluate_healing(*args, use_graph=False)
        with_graph = evaluate_healing(*args)
        assert PRIORITY_ORDER.index(with_graph["priority"]) < PRIORITY_ORDER.index(rules_only["priority"])
        assert any("Compounding risk" in a for a in with_graph["alerts"])

    def test_graph_never_lowers_rule_priority(self):
        tissue = {"red": 60.0, "yellow": 20.0, "black": 20.0}
        result = evaluate_healing(HEALING_DELTA, tissue, _healthy_patient())
        assert result["priority"] == "CRITICAL"

    def test_healthy_healing_patient_stays_ok(self):
        result = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient())
        assert result["priority"] == "OK"
        assert not any("Compounding risk" in a for a in result["alerts"])

    def test_risk_score_is_returned_and_ranks_patients(self):
        low = evaluate_healing(HEALING_DELTA, HEALTHY_TISSUE, _healthy_patient())
        high = evaluate_healing(0.3, STALLED_NECROTIC["tissue_ratios"], HIGH_RISK_PATIENT)
        assert high["risk_score"] > low["risk_score"] >= 0.0
