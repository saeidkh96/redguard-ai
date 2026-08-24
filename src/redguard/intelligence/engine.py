from __future__ import annotations

from redguard.intelligence.evidence import EvidenceBuilder
from redguard.intelligence.models import InspectionFinding
from redguard.intelligence.risk import InspectionRiskEngine


class InspectionIntelligenceEngine:
    """Generate an explainable component-level inspection finding."""

    def __init__(self) -> None:
        self.evidence_builder = EvidenceBuilder()
        self.risk_engine = InspectionRiskEngine()

    def inspect(
        self,
        *,
        component_id: str,
        component_type: str,
        change_score: float,
        fingerprint_similarity: float,
        anomaly_score: float,
        edge_difference: float,
        texture_difference: float,
        reference_consensus: float,
    ) -> InspectionFinding:

        if not component_id:
            raise ValueError("component_id must not be empty")

        if not component_type:
            raise ValueError("component_type must not be empty")

        evidence = self.evidence_builder.build(
            change_score=change_score,
            fingerprint_similarity=fingerprint_similarity,
            anomaly_score=anomaly_score,
            edge_difference=edge_difference,
            texture_difference=texture_difference,
            reference_consensus=reference_consensus,
        )

        risk, decision, severity, confidence = self.risk_engine.calculate(
            evidence
        )

        abnormal = [item for item in evidence if item.abnormal]

        if abnormal:
            reasons = "; ".join(item.description for item in abnormal)

            explanation = (
                f"{component_id} is inconsistent with its expected "
                f"physical state. Evidence: {reasons}."
            )
        else:
            explanation = (
                f"{component_id} is consistent with its known-normal "
                "visual state."
            )

        return InspectionFinding(
            component_id=component_id,
            component_type=component_type,
            decision=decision,
            severity=severity,
            risk_score=risk,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
        )