from dataclasses import dataclass

import numpy as np

from redguard.features.backbone import (
    VisionBackbone,
)


@dataclass(frozen=True)
class FingerprintComparison:
    similarity: float
    distance: float
    same_instance_candidate: bool


class ComponentFingerprinter:
    """Generate and compare visual component fingerprints."""

    def __init__(
        self,
        backbone: VisionBackbone | None = None,
        similarity_threshold: float = 0.97,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        self.backbone = (
            backbone
            or VisionBackbone()
        )

        self.similarity_threshold = (
            similarity_threshold
        )

    def fingerprint(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return self.backbone.global_embedding(
            image
        )

    def compare(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> FingerprintComparison:
        reference_embedding = (
            self.fingerprint(reference)
        )

        inspection_embedding = (
            self.fingerprint(inspection)
        )

        similarity = float(
            np.dot(
                reference_embedding,
                inspection_embedding,
            )
        )

        similarity = float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

        distance = 1.0 - similarity

        return FingerprintComparison(
            similarity=similarity,
            distance=distance,
            same_instance_candidate=(
                similarity
                >= self.similarity_threshold
            ),
        )
