import numpy as np

from redguard.inspection.multi_reference import (
    MultiReferenceVerifier,
    VerificationDecision,
)
from redguard.reference import ReferenceSample, ReferenceSet


def normalized(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def build_reference_set() -> ReferenceSet:
    reference_set = ReferenceSet(
        component_id="Q14",
        component_type="transistor",
    )

    references = [
        ("normal_light", [1.00, 0.02, 0.01, 0.00]),
        ("normal_dark", [0.99, 0.04, 0.01, 0.01]),
        ("normal_view", [1.00, -0.02, 0.03, 0.01]),
    ]

    for sample_id, values in references:
        reference_set.add(
            ReferenceSample(
                sample_id=sample_id,
                component_id="Q14",
                component_type="transistor",
                fingerprint=normalized(values),
            )
        )

    return reference_set


def test_normal_component_passes_multi_reference_verification():
    verifier = MultiReferenceVerifier(
        similarity_threshold=0.97,
        pass_consensus=0.67,
    )

    result = verifier.verify(
        normalized([0.995, 0.025, 0.015, 0.005]),
        build_reference_set(),
    )

    assert result.decision is VerificationDecision.PASS
    assert result.match.consensus_ratio >= 0.67
    assert result.match.best_similarity >= 0.97
    assert result.risk_score < 0.25


def test_replacement_component_fails_multi_reference_verification():
    verifier = MultiReferenceVerifier(
        similarity_threshold=0.97,
        pass_consensus=0.67,
        review_consensus=0.34,
    )

    result = verifier.verify(
        normalized([0.60, 0.65, 0.40, 0.20]),
        build_reference_set(),
    )

    assert result.decision is VerificationDecision.FAIL
    assert result.match.consensus_ratio == 0.0
    assert result.risk_score > 0.50