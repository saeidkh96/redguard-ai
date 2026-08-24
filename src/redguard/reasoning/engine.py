from __future__ import annotations

from dataclasses import dataclass

from redguard.intelligence.models import InspectionFinding
from redguard.reasoning.base import ReasoningProvider
from redguard.reasoning.context import build_reasoning_context
from redguard.reasoning.models import ReasoningOutput


@dataclass(slots=True)
class ReasonedInspectionFinding:
    finding: InspectionFinding
    reasoning: ReasoningOutput


class VisionReasoningEngine:
    """
    Adds contextual explanation to deterministic inspection findings.

    Safety contract:
    - decision is immutable
    - severity is immutable
    - risk score is immutable
    - confidence is immutable

    The reasoner may only generate explanation and observations.
    """

    def __init__(
        self,
        provider: ReasoningProvider,
    ) -> None:
        self.provider = provider

    def reason(
        self,
        finding: InspectionFinding,
    ) -> ReasonedInspectionFinding:
        original_decision = finding.decision
        original_severity = finding.severity
        original_risk = finding.risk_score
        original_confidence = finding.confidence

        context = build_reasoning_context(
            finding
        )

        reasoning = self.provider.explain(
            context
        )

        if finding.decision is not original_decision:
            raise RuntimeError(
                "reasoning provider modified inspection decision"
            )

        if finding.severity is not original_severity:
            raise RuntimeError(
                "reasoning provider modified severity"
            )

        if finding.risk_score != original_risk:
            raise RuntimeError(
                "reasoning provider modified risk score"
            )

        if finding.confidence != original_confidence:
            raise RuntimeError(
                "reasoning provider modified confidence"
            )

        return ReasonedInspectionFinding(
            finding=finding,
            reasoning=reasoning,
        )