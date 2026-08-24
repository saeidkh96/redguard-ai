from __future__ import annotations

from redguard.intelligence.models import InspectionFinding
from redguard.reasoning.models import ReasoningContext


def build_reasoning_context(
    finding: InspectionFinding,
) -> ReasoningContext:
    evidence = tuple(
        {
            "type": item.evidence_type.value,
            "score": float(item.score),
            "weight": float(item.weight),
            "abnormal": bool(item.abnormal),
            "description": item.description,
        }
        for item in finding.evidence
    )

    return ReasoningContext(
        component_id=finding.component_id,
        component_type=finding.component_type,
        decision=finding.decision.value,
        severity=finding.severity.value,
        risk_score=float(finding.risk_score),
        confidence=float(finding.confidence),
        evidence=evidence,
    )