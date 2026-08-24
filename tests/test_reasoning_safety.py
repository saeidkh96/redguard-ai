from redguard.intelligence import (
    InspectionIntelligenceEngine,
)
from redguard.reasoning import (
    RuleBasedReasoner,
    VisionReasoningEngine,
)


def build_finding():
    return (
        InspectionIntelligenceEngine()
        .inspect(
            component_id="Q14",
            component_type="transistor",
            change_score=0.90,
            fingerprint_similarity=0.30,
            anomaly_score=0.95,
            edge_difference=0.75,
            texture_difference=0.70,
            reference_consensus=0.0,
        )
    )


def test_reasoning_does_not_change_decision():
    finding = build_finding()

    before = finding.decision

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert result.finding.decision is before


def test_reasoning_does_not_change_risk():
    finding = build_finding()

    before = finding.risk_score

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert (
        result.finding.risk_score
        == before
    )


def test_reasoning_does_not_change_severity():
    finding = build_finding()

    before = finding.severity

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert result.finding.severity is before


def test_reasoning_does_not_change_confidence():
    finding = build_finding()

    before = finding.confidence

    result = VisionReasoningEngine(
        RuleBasedReasoner()
    ).reason(
        finding
    )

    assert (
        result.finding.confidence
        == before
    )