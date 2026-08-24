from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ReferenceSample:
    """One known-normal visual observation of a physical component."""

    sample_id: str
    component_id: str
    component_type: str
    fingerprint: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fingerprint = np.asarray(self.fingerprint, dtype=np.float32).reshape(-1)

        if not self.sample_id:
            raise ValueError("sample_id must not be empty")

        if not self.component_id:
            raise ValueError("component_id must not be empty")

        if not self.component_type:
            raise ValueError("component_type must not be empty")

        if fingerprint.size == 0:
            raise ValueError("fingerprint must not be empty")

        if not np.all(np.isfinite(fingerprint)):
            raise ValueError("fingerprint must contain only finite values")

        self.fingerprint = fingerprint


@dataclass(slots=True)
class ReferenceSet:
    """Known-normal reference population for one physical component."""

    component_id: str
    component_type: str
    samples: list[ReferenceSample] = field(default_factory=list)

    def add(self, sample: ReferenceSample) -> None:
        if sample.component_id != self.component_id:
            raise ValueError(
                f"component mismatch: expected {self.component_id}, "
                f"got {sample.component_id}"
            )

        if sample.component_type != self.component_type:
            raise ValueError(
                f"component type mismatch: expected {self.component_type}, "
                f"got {sample.component_type}"
            )

        if any(existing.sample_id == sample.sample_id for existing in self.samples):
            raise ValueError(f"duplicate reference sample: {sample.sample_id}")

        if self.samples:
            expected_dimension = self.samples[0].fingerprint.shape[0]
            if sample.fingerprint.shape[0] != expected_dimension:
                raise ValueError(
                    "reference fingerprints must have the same dimension"
                )

        self.samples.append(sample)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def fingerprints(self) -> np.ndarray:
        if not self.samples:
            raise ValueError("reference set is empty")

        return np.stack(
            [sample.fingerprint for sample in self.samples],
            axis=0,
        )


@dataclass(slots=True, frozen=True)
class ReferenceMatch:
    sample_id: str
    similarity: float
    accepted: bool


@dataclass(slots=True)
class ReferenceMatchResult:
    component_id: str
    reference_count: int
    best_sample_id: str
    best_similarity: float
    mean_similarity: float
    median_similarity: float
    consensus_ratio: float
    accepted_count: int
    matches: list[ReferenceMatch]

    @property
    def has_consensus(self) -> bool:
        return self.consensus_ratio >= 0.5