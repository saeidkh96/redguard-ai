from redguard.intelligence import EvidenceBuilder, EvidenceType


def test_evidence_builder_generates_all_signals():
    evidence = EvidenceBuilder().build(
        change_score=0.8,
        fingerprint_similarity=0.7,
        anomaly_score=0.9,
        edge_difference=0.3,
        texture_difference=0.2,
        reference_consensus=0.0,
    )

    assert len(evidence) == 6

    types = {item.evidence_type for item in evidence}

    assert EvidenceType.VISUAL_CHANGE in types
    assert EvidenceType.FINGERPRINT in types
    assert EvidenceType.ANOMALY in types
    assert EvidenceType.EDGE in types
    assert EvidenceType.TEXTURE in types
    assert EvidenceType.REFERENCE_CONSENSUS in types


def test_normal_signals_are_not_abnormal():
    evidence = EvidenceBuilder().build(
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.05,
        edge_difference=0.02,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    assert not any(item.abnormal for item in evidence)


def test_altered_signals_generate_abnormal_evidence():
    evidence = EvidenceBuilder().build(
        change_score=0.8,
        fingerprint_similarity=0.70,
        anomaly_score=0.90,
        edge_difference=0.30,
        texture_difference=0.25,
        reference_consensus=0.0,
    )

    assert sum(item.abnormal for item in evidence) >= 4