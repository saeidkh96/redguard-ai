import numpy as np
import pytest

from redguard.reference import (
    MultiReferenceMatcher,
    ReferenceSample,
    ReferenceSet,
)


def normalized(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def build_reference_set() -> ReferenceSet:
    reference_set = ReferenceSet(
        component_id="Q14",
        component_type="transistor",
    )

    reference_set.add(
        ReferenceSample(
            sample_id="normal_001",
            component_id="Q14",
            component_type="transistor",
            fingerprint=normalized([1.0, 0.02, 0.01]),
        )
    )

    reference_set.add(
        ReferenceSample(
            sample_id="normal_002",
            component_id="Q14",
            component_type="transistor",
            fingerprint=normalized([0.99, 0.04, 0.01]),
        )
    )

    reference_set.add(
        ReferenceSample(
            sample_id="normal_003",
            component_id="Q14",
            component_type="transistor",
            fingerprint=normalized([1.0, -0.02, 0.03]),
        )
    )

    return reference_set


def test_matcher_accepts_normal_like_fingerprint():
    matcher = MultiReferenceMatcher(similarity_threshold=0.97)

    result = matcher.match(
        normalized([0.995, 0.025, 0.015]),
        build_reference_set(),
    )

    assert result.reference_count == 3
    assert result.best_similarity >= 0.99
    assert result.consensus_ratio == pytest.approx(1.0)
    assert result.accepted_count == 3


def test_matcher_rejects_replacement_like_fingerprint():
    matcher = MultiReferenceMatcher(similarity_threshold=0.97)

    result = matcher.match(
        normalized([0.75, 0.60, 0.20]),
        build_reference_set(),
    )

    assert result.best_similarity < 0.97
    assert result.consensus_ratio == pytest.approx(0.0)
    assert result.accepted_count == 0


def test_matcher_reports_best_reference():
    matcher = MultiReferenceMatcher(similarity_threshold=0.97)

    result = matcher.match(
        normalized([0.99, 0.04, 0.01]),
        build_reference_set(),
    )

    assert result.best_sample_id == "normal_002"


def test_matcher_rejects_dimension_mismatch():
    matcher = MultiReferenceMatcher()

    with pytest.raises(ValueError):
        matcher.match(
            np.array([1.0, 0.0], dtype=np.float32),
            build_reference_set(),
        )