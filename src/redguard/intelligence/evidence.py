from __future__ import annotations

import numpy as np

from redguard.intelligence.models import EvidenceType, InspectionEvidence


class EvidenceBuilder:
    """Convert inspection signals into normalized evidence."""

    def build(
        self,
        *,
        change_score: float,
        fingerprint_similarity: float,
        anomaly_score: float,
        edge_difference: float,
        texture_difference: float,
        reference_consensus: float,
    ) -> list[InspectionEvidence]:

        change_score = self._unit(change_score)
        fingerprint_similarity = self._unit(fingerprint_similarity)
        anomaly_score = self._unit(anomaly_score)
        edge_difference = self._unit(edge_difference)
        texture_difference = self._unit(texture_difference)
        reference_consensus = self._unit(reference_consensus)

        fingerprint_risk = 1.0 - fingerprint_similarity
        consensus_risk = 1.0 - reference_consensus

        return [
            InspectionEvidence(
                evidence_type=EvidenceType.VISUAL_CHANGE,
                score=change_score,
                weight=0.15,
                description="Localized visual change detected",
                abnormal=change_score >= 0.20,
            ),
            InspectionEvidence(
                evidence_type=EvidenceType.FINGERPRINT,
                score=fingerprint_risk,
                weight=0.25,
                description="Component visual identity mismatch",
                abnormal=fingerprint_similarity < 0.97,
            ),
            InspectionEvidence(
                evidence_type=EvidenceType.ANOMALY,
                score=anomaly_score,
                weight=0.25,
                description="Strong local visual anomaly detected",
                abnormal=anomaly_score >= 0.25,
            ),
            InspectionEvidence(
                evidence_type=EvidenceType.EDGE,
                score=edge_difference,
                weight=0.10,
                description="Component edge structure differs from reference",
                abnormal=edge_difference >= 0.15,
            ),
            InspectionEvidence(
                evidence_type=EvidenceType.TEXTURE,
                score=texture_difference,
                weight=0.10,
                description="Component texture differs from reference",
                abnormal=texture_difference >= 0.15,
            ),
            InspectionEvidence(
                evidence_type=EvidenceType.REFERENCE_CONSENSUS,
                score=consensus_risk,
                weight=0.15,
                description="Known-normal reference consensus failed",
                abnormal=reference_consensus < 0.67,
            ),
        ]

    @staticmethod
    def _unit(value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("inspection signal must be finite")

        return float(np.clip(value, 0.0, 1.0))