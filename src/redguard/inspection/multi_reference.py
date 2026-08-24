from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from redguard.reference.matcher import MultiReferenceMatcher
from redguard.reference.models import ReferenceMatchResult, ReferenceSet


class VerificationDecision(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass(slots=True)
class MultiReferenceVerificationResult:
    component_id: str
    decision: VerificationDecision
    risk_score: float
    confidence: float
    match: ReferenceMatchResult


class MultiReferenceVerifier:
    """Convert reference-population evidence into an inspection decision."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.97,
        pass_consensus: float = 0.67,
        review_consensus: float = 0.34,
    ) -> None:
        if not 0.0 <= review_consensus <= pass_consensus <= 1.0:
            raise ValueError(
                "consensus thresholds must satisfy "
                "0 <= review <= pass <= 1"
            )

        self.matcher = MultiReferenceMatcher(
            similarity_threshold=similarity_threshold,
        )
        self.similarity_threshold = float(similarity_threshold)
        self.pass_consensus = float(pass_consensus)
        self.review_consensus = float(review_consensus)

    def verify(
        self,
        inspection_fingerprint: np.ndarray,
        reference_set: ReferenceSet,
    ) -> MultiReferenceVerificationResult:
        match = self.matcher.match(
            inspection_fingerprint,
            reference_set,
        )

        similarity_risk = np.clip(
            (self.similarity_threshold - match.mean_similarity)
            / max(self.similarity_threshold, 1e-12),
            0.0,
            1.0,
        )

        consensus_risk = 1.0 - match.consensus_ratio

        risk_score = float(
            np.clip(
                (0.60 * consensus_risk)
                + (0.40 * similarity_risk),
                0.0,
                1.0,
            )
        )

        if (
            match.consensus_ratio >= self.pass_consensus
            and match.best_similarity >= self.similarity_threshold
        ):
            decision = VerificationDecision.PASS

        elif (
            match.consensus_ratio >= self.review_consensus
            or match.best_similarity >= self.similarity_threshold
        ):
            decision = VerificationDecision.REVIEW

        else:
            decision = VerificationDecision.FAIL

        confidence = self._confidence(
            match=match,
            decision=decision,
        )

        return MultiReferenceVerificationResult(
            component_id=reference_set.component_id,
            decision=decision,
            risk_score=risk_score,
            confidence=confidence,
            match=match,
        )

    def _confidence(
        self,
        *,
        match: ReferenceMatchResult,
        decision: VerificationDecision,
    ) -> float:
        if decision is VerificationDecision.PASS:
            confidence = (
                0.5 * match.consensus_ratio
                + 0.5 * max(match.best_similarity, 0.0)
            )

        elif decision is VerificationDecision.FAIL:
            confidence = (
                0.5 * (1.0 - match.consensus_ratio)
                + 0.5 * (1.0 - max(match.best_similarity, 0.0))
            )

        else:
            confidence = 1.0 - abs(match.consensus_ratio - 0.5)

        return float(np.clip(confidence, 0.0, 1.0))