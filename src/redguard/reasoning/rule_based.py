from __future__ import annotations

from redguard.reasoning.base import ReasoningProvider
from redguard.reasoning.models import (
    ReasoningContext,
    ReasoningOutput,
)


class RuleBasedReasoner(ReasoningProvider):
    """
    Deterministic reasoning provider used as a safe baseline.

    It explains structured evidence without modifying
    deterministic inspection decisions.
    """

    def explain(
        self,
        context: ReasoningContext,
    ) -> ReasoningOutput:
        abnormal = [
            item
            for item in context.evidence
            if item["abnormal"]
        ]

        observations = tuple(
            str(item["description"])
            for item in abnormal
        )

        if context.decision == "PASS":
            explanation = (
                f"{context.component_id} is consistent with its "
                f"known-normal visual state. No significant abnormal "
                f"evidence was identified."
            )

        elif context.decision == "REVIEW":
            evidence_text = self._join_observations(
                observations
            )

            explanation = (
                f"{context.component_id} requires review. "
                f"The deterministic inspection system found "
                f"moderate or conflicting evidence"
                f"{evidence_text}."
            )

        else:
            evidence_text = self._join_observations(
                observations
            )

            explanation = (
                f"{context.component_id} is inconsistent with its "
                f"expected physical state. Multiple inspection signals "
                f"support the deterministic FAIL decision"
                f"{evidence_text}."
            )

        return ReasoningOutput(
            explanation=explanation,
            observations=observations,
            provider="rule-based",
        )

    @staticmethod
    def _join_observations(
        observations: tuple[str, ...],
    ) -> str:
        if not observations:
            return ""

        return ": " + "; ".join(
            observations
        )