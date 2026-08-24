from redguard.intelligence import (
    InspectionDecision,
    InspectionIntelligenceEngine,
    Severity,
)


def test_normal_component_generates_explainable_pass():
    engine = InspectionIntelligenceEngine()

    finding = engine.inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.05,
        edge_difference=0.02,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    assert finding.decision is InspectionDecision.PASS
    assert finding.risk_score < 0.20
    assert len(finding.abnormal_evidence) == 0
    assert "known-normal" in finding.explanation


def test_altered_component_generates_explainable_fail():
    engine = InspectionIntelligenceEngine()

    finding = engine.inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.35,
        anomaly_score=0.95,
        edge_difference=0.70,
        texture_difference=0.65,
        reference_consensus=0.0,
    )

    assert finding.decision is InspectionDecision.FAIL
    assert finding.severity in {Severity.HIGH, Severity.CRITICAL}
    assert finding.risk_score >= 0.45
    assert len(finding.abnormal_evidence) >= 4
    assert "Q14" in finding.explanation