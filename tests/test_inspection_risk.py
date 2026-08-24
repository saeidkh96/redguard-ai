from redguard.intelligence import (
    EvidenceBuilder,
    InspectionDecision,
    InspectionRiskEngine,
    Severity,
)


def test_normal_evidence_produces_pass():
    evidence = EvidenceBuilder().build(
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.05,
        edge_difference=0.02,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    risk, decision, severity, confidence = (
        InspectionRiskEngine().calculate(evidence)
    )

    assert risk < 0.20
    assert decision is InspectionDecision.PASS
    assert severity in {Severity.NONE, Severity.LOW}
    assert 0.0 <= confidence <= 1.0


def test_strong_abnormal_evidence_produces_fail():
    evidence = EvidenceBuilder().build(
        change_score=0.95,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.80,
        texture_difference=0.75,
        reference_consensus=0.0,
    )

    risk, decision, severity, confidence = (
        InspectionRiskEngine().calculate(evidence)
    )

    assert risk >= 0.45
    assert decision is InspectionDecision.FAIL
    assert severity in {Severity.HIGH, Severity.CRITICAL}
    assert 0.0 <= confidence <= 1.0