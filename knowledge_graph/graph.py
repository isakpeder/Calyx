"""
graph.py — ChroniScan Clinical Knowledge Graph.

Encodes evidence-based relationships between patient comorbidities,
biomarker states, and wound healing outcomes as a directed NetworkX graph.
Edge weights represent clinical evidence strength (0.0–1.0).
"""

from __future__ import annotations

import networkx as nx

# ---------------------------------------------------------------------------
# Node taxonomy constants
# ---------------------------------------------------------------------------

CONDITION_NODES = [
    "Type_2_Diabetes",
    "Obesity",
    "Hypertension",
    "Peripheral_Artery_Disease",
    "Malnutrition",
    "Low_Mobility",
]

BIOMARKER_NODES = [
    "Hyperglycemia",        # blood_glucose > 180 mg/dL
    "Low_Serum_Albumin",    # serum_albumin < 3.0 g/dL
    "High_BMI",             # proxy for Obesity comorbidity
    "Poor_Perfusion",       # proxy for Peripheral Artery Disease
]

WOUND_STATE_NODES = [
    "Necrotic_Tissue",      # black tissue > 5% (from computer vision)
    "Slough_Present",       # yellow tissue > 10%
    "Non_Healing_Wound",    # area not shrinking since last scan
]

OUTCOME_NODES = [
    "Wound_Stagnation",
    "Infection_Risk",
    "Delayed_Healing",
    "Necrosis_Risk",
]

# Clinical severity of each outcome, used to collapse outcome evidence
# into a single patient risk score.
OUTCOME_SEVERITY: dict[str, float] = {
    "Necrosis_Risk":    1.00,
    "Infection_Risk":   0.80,
    "Wound_Stagnation": 0.60,
    "Delayed_Healing":  0.50,
}

# Maps human-readable comorbidity strings (from patient profiles) → graph node IDs
COMORBIDITY_TO_NODE: dict[str, str] = {
    "Type 2 Diabetes":           "Type_2_Diabetes",
    "Obesity":                   "Obesity",
    "Hypertension":              "Hypertension",
    "Peripheral Artery Disease": "Peripheral_Artery_Disease",
    "Malnutrition":              "Malnutrition",
    "Low_Mobility":              "Low_Mobility",
}

# Module-level cache — graph is built once and reused
_GRAPH_CACHE: nx.DiGraph | None = None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build() -> nx.DiGraph:
    """Internal builder — constructs the clinical knowledge DiGraph."""
    G = nx.DiGraph()

    # Add condition nodes
    for node in CONDITION_NODES:
        G.add_node(node, type="condition", description=node.replace("_", " "))

    # Add biomarker nodes
    for node in BIOMARKER_NODES:
        G.add_node(node, type="biomarker", description=node.replace("_", " "))

    # Add wound state nodes
    for node in WOUND_STATE_NODES:
        G.add_node(node, type="wound_state", description=node.replace("_", " "))

    # Add outcome nodes
    for node in OUTCOME_NODES:
        G.add_node(node, type="outcome", description=node.replace("_", " "))

    # ------------------------------------------------------------------
    # Edges: Conditions → Biomarkers
    # ------------------------------------------------------------------
    G.add_edge("Type_2_Diabetes", "Hyperglycemia",
               weight=0.95, label="causes_chronic_hyperglycemia")
    G.add_edge("Type_2_Diabetes", "Low_Serum_Albumin",
               weight=0.60, label="impairs_protein_synthesis")
    G.add_edge("Obesity", "High_BMI",
               weight=1.00, label="defines")
    G.add_edge("Obesity", "Low_Serum_Albumin",
               weight=0.45, label="associated_with_malnutrition")
    G.add_edge("Malnutrition", "Low_Serum_Albumin",
               weight=0.95, label="directly_causes")
    G.add_edge("Peripheral_Artery_Disease", "Poor_Perfusion",
               weight=0.90, label="reduces_tissue_oxygenation")

    # ------------------------------------------------------------------
    # Edges: Conditions → Outcomes (direct, strong evidence)
    # ------------------------------------------------------------------
    G.add_edge("Low_Mobility", "Wound_Stagnation",
               weight=0.70, label="reduces_offloading_and_perfusion")
    G.add_edge("Hypertension", "Wound_Stagnation",
               weight=0.40, label="microvascular_disease")
    G.add_edge("Type_2_Diabetes", "Delayed_Healing",
               weight=0.80, label="multi_mechanism_impairment")

    # ------------------------------------------------------------------
    # Edges: Biomarkers → Outcomes
    # ------------------------------------------------------------------
    G.add_edge("Hyperglycemia", "Wound_Stagnation",
               weight=0.85, label="impairs_neutrophil_function")
    G.add_edge("Hyperglycemia", "Infection_Risk",
               weight=0.80, label="promotes_bacterial_growth")
    G.add_edge("Hyperglycemia", "Delayed_Healing",
               weight=0.85, label="inhibits_collagen_synthesis")
    G.add_edge("Low_Serum_Albumin", "Delayed_Healing",
               weight=0.80, label="insufficient_tissue_repair_substrate")
    G.add_edge("Low_Serum_Albumin", "Wound_Stagnation",
               weight=0.75, label="reduces_oncotic_pressure")
    G.add_edge("High_BMI", "Wound_Stagnation",
               weight=0.65, label="increases_wound_tension")
    G.add_edge("High_BMI", "Infection_Risk",
               weight=0.60, label="adipose_tissue_hypoxia")
    G.add_edge("Poor_Perfusion", "Necrosis_Risk",
               weight=0.90, label="tissue_ischemia")
    G.add_edge("Poor_Perfusion", "Delayed_Healing",
               weight=0.85, label="insufficient_oxygen_delivery")

    # ------------------------------------------------------------------
    # Edges: Wound state (computer vision) → Outcomes
    # ------------------------------------------------------------------
    G.add_edge("Necrotic_Tissue", "Necrosis_Risk",
               weight=0.90, label="devitalized_tissue_present")
    G.add_edge("Necrotic_Tissue", "Infection_Risk",
               weight=0.60, label="bacterial_reservoir")
    G.add_edge("Slough_Present", "Infection_Risk",
               weight=0.55, label="biofilm_substrate")
    G.add_edge("Slough_Present", "Wound_Stagnation",
               weight=0.60, label="blocks_granulation")
    G.add_edge("Non_Healing_Wound", "Wound_Stagnation",
               weight=0.85, label="observed_stall")
    G.add_edge("Non_Healing_Wound", "Delayed_Healing",
               weight=0.60, label="observed_stall")

    return G


def build_graph() -> nx.DiGraph:
    """
    Construct and return the clinical knowledge DiGraph.
    Results are cached — the graph is built only once per process.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = _build()
    return _GRAPH_CACHE


# ---------------------------------------------------------------------------
# Patient-specific graph traversal
# ---------------------------------------------------------------------------

def _active_nodes(patient_data: dict, wound: dict | None, G: nx.DiGraph) -> set[str]:
    """Map a patient's comorbidities, biomarkers and wound state to graph nodes."""
    comorbidities  = patient_data.get("comorbidities", [])
    blood_glucose  = patient_data.get("blood_glucose", 0.0)
    serum_albumin  = patient_data.get("serum_albumin", 4.0)
    mobility_score = patient_data.get("mobility_score", 10)

    active: set[str] = set()

    for comorbidity in comorbidities:
        node_id = COMORBIDITY_TO_NODE.get(comorbidity)
        if node_id and node_id in G:
            active.add(node_id)

    if blood_glucose > 180:
        active.add("Hyperglycemia")
    if serum_albumin < 3.0:
        active.add("Low_Serum_Albumin")
    if "Obesity" in comorbidities:
        active.add("High_BMI")
    if "Peripheral Artery Disease" in comorbidities:
        active.add("Poor_Perfusion")
    if mobility_score < 4:
        active.add("Low_Mobility")

    if wound is not None:
        tissue = wound.get("tissue_ratios", {})
        if tissue.get("black", 0.0) > 5:
            active.add("Necrotic_Tissue")
        if tissue.get("yellow", 0.0) > 10:
            active.add("Slough_Present")
        if wound.get("area_delta", -1.0) >= 0:
            active.add("Non_Healing_Wound")

    return active


def get_outcome_scores(
    patient_data: dict,
    wound: dict | None = None,
    G: nx.DiGraph | None = None,
) -> dict[str, float]:
    """
    Score every reachable OUTCOME node for a patient.

    Algorithm:
    1. Map comorbidities, biomarker thresholds and (optionally) wound state
       from computer vision → active graph nodes (activation 1.0)
    2. Multi-source BFS from all active nodes at once (max depth 2). Each
       intermediate node is activated once, by its strongest parent at the
       depth it is first reached, with activation = parent activation × edge
       weight — so a mechanism such as Type 2 Diabetes → Hyperglycemia is
       never counted twice
    3. Each outcome sums evidence (activation × edge weight) from every
       activated parent, so independent risk factors compound

    Parameters
    ----------
    patient_data : dict
        Patient profile dict from mock_patients.py
    wound : dict, optional
        {"tissue_ratios": {...}, "area_delta": float} from the latest scan
    G : nx.DiGraph, optional
        Pre-built graph (uses cached build_graph() if omitted)

    Returns
    -------
    dict[str, float]
        Outcome node ID → accumulated evidence (≥ 0)
    """
    if G is None:
        G = build_graph()

    active = _active_nodes(patient_data, wound, G)
    activation = {node: 1.0 for node in active}
    depth = {node: 0 for node in active}

    # Step 2: multi-source BFS over non-outcome nodes, one depth level at a
    # time. A node reached by several parents at the same depth takes the
    # strongest path, so the result never depends on set iteration order.
    frontier = sorted(active)
    for level in range(1, 3):
        discovered: dict[str, float] = {}
        for current in frontier:
            for neighbor in G.successors(current):
                if G.nodes[neighbor].get("type") == "outcome" or neighbor in activation:
                    continue
                strength = activation[current] * G[current][neighbor].get("weight", 0.5)
                discovered[neighbor] = max(discovered.get(neighbor, 0.0), strength)
        activation.update(discovered)
        depth.update({node: level for node in discovered})
        frontier = sorted(discovered)

    # Step 3: outcomes accumulate evidence from every activated parent
    evidence: dict[str, float] = {}
    for node, strength in activation.items():
        if depth[node] >= 2:
            continue
        for neighbor in G.successors(node):
            if G.nodes[neighbor].get("type") == "outcome":
                weight = G[node][neighbor].get("weight", 0.5)
                evidence[neighbor] = evidence.get(neighbor, 0.0) + strength * weight

    return {node: round(score, 4) for node, score in evidence.items()}


def compute_risk_score(outcome_scores: dict[str, float]) -> float:
    """
    Collapse outcome evidence into one patient risk score.

    Each outcome's evidence is scaled by its clinical severity and summed.
    The score is unbounded so compounding risk keeps raising it rather than
    saturating; GRAPH_PRIORITY_THRESHOLDS in reasoning.py map it to a priority.
    """
    total = sum(OUTCOME_SEVERITY.get(node, 0.5) * ev for node, ev in outcome_scores.items())
    return round(total, 4)


def get_risk_factors(
    patient_data: dict,
    G: nx.DiGraph | None = None,
    wound: dict | None = None,
) -> list[str]:
    """Return reachable OUTCOME node IDs, strongest evidence first."""
    scores = get_outcome_scores(patient_data, wound, G)
    return sorted(scores, key=scores.get, reverse=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_graph_summary(G: nx.DiGraph) -> dict:
    """Return a summary dict of the knowledge graph for debugging/display."""
    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "condition_nodes": [n for n, d in G.nodes(data=True) if d.get("type") == "condition"],
        "biomarker_nodes": [n for n, d in G.nodes(data=True) if d.get("type") == "biomarker"],
        "wound_state_nodes": [n for n, d in G.nodes(data=True) if d.get("type") == "wound_state"],
        "outcome_nodes":   [n for n, d in G.nodes(data=True) if d.get("type") == "outcome"],
    }
