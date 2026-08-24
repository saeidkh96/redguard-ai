from redguard.intelligence import (
    InspectionIntelligenceEngine,
)
from redguard.reasoning import (
    build_reasoning_context,
)


def test_reasoning_context_contains_inspection_state():
    finding = InspectionIntelligenceEngine().inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.75,
        texture_difference=0.70,
        reference_consensus=0.0,
    )

    context = build_reasoning_context(
        finding
    )

    assert context.component_id == "Q14"
    assert context.component_type == "transistor"
    assert context.decision == "FAIL"
    assert context.severity in {
        "HIGH",
        "CRITICAL",
    }

    assert len(context.evidence) == 6


def test_reasoning_context_preserves_evidence():
    finding = InspectionIntelligenceEngine().inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.75,
        texture_difference=0.70,
        reference_consensus=0.0,
    )

    context = build_reasoning_context(
        finding
    )

    abnormal = [
        item
        for item in context.evidence
        if item["abnormal"]
    ]

    assert len(abnormal) >= 4