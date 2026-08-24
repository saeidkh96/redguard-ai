from redguard.intelligence import (
    InspectionDecision,
    InspectionIntelligenceEngine,
)
from redguard.reasoning import (
    RuleBasedReasoner,
    VisionReasoningEngine,
)


def test_reasoning_engine_explains_fail():
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

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert (
        result.finding.decision
        is InspectionDecision.FAIL
    )

    assert "Q14" in (
        result.reasoning.explanation
    )

    assert (
        len(
            result.reasoning.observations
        )
        >= 4
    )

    assert (
        result.reasoning.provider
        == "rule-based"
    )


def test_reasoning_engine_explains_pass():
    finding = InspectionIntelligenceEngine().inspect(
        component_id="R27",
        component_type="resistor",
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.04,
        edge_difference=0.01,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert (
        result.finding.decision
        is InspectionDecision.PASS
    )

    assert (
        "known-normal"
        in result.reasoning.explanation
    )

    assert (
        result.reasoning.observations
        == ()
    )