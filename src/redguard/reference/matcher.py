from __future__ import annotations

import numpy as np

from redguard.reference.models import (
    ReferenceMatch,
    ReferenceMatchResult,
    ReferenceSet,
)


class MultiReferenceMatcher:
    """Compare one inspection fingerprint against known-normal references."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.97,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.similarity_threshold = float(similarity_threshold)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)

        if vector.size == 0:
            raise ValueError("fingerprint must not be empty")

        if not np.all(np.isfinite(vector)):
            raise ValueError("fingerprint must contain only finite values")

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            raise ValueError("fingerprint norm must be greater than zero")

        return vector / norm

    def match(
        self,
        inspection_fingerprint: np.ndarray,
        reference_set: ReferenceSet,
    ) -> ReferenceMatchResult:
        if reference_set.size == 0:
            raise ValueError("reference set must contain at least one sample")

        query = self._normalize(inspection_fingerprint)

        reference_vectors = reference_set.fingerprints

        if reference_vectors.shape[1] != query.shape[0]:
            raise ValueError(
                "inspection fingerprint dimension does not match references"
            )

        normalized_references = np.stack(
            [self._normalize(vector) for vector in reference_vectors],
            axis=0,
        )

        similarities = normalized_references @ query

        similarities = np.clip(
            similarities.astype(np.float64),
            -1.0,
            1.0,
        )

        matches = [
            ReferenceMatch(
                sample_id=sample.sample_id,
                similarity=float(similarity),
                accepted=bool(similarity >= self.similarity_threshold),
            )
            for sample, similarity in zip(
                reference_set.samples,
                similarities,
                strict=True,
            )
        ]

        best_index = int(np.argmax(similarities))
        accepted_count = sum(match.accepted for match in matches)

        return ReferenceMatchResult(
            component_id=reference_set.component_id,
            reference_count=reference_set.size,
            best_sample_id=reference_set.samples[best_index].sample_id,
            best_similarity=float(similarities[best_index]),
            mean_similarity=float(np.mean(similarities)),
            median_similarity=float(np.median(similarities)),
            consensus_ratio=float(accepted_count / reference_set.size),
            accepted_count=accepted_count,
            matches=matches,
        )