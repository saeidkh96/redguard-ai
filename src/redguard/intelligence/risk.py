from __future__ import annotations

import numpy as np

from redguard.intelligence.models import (
    InspectionDecision,
    InspectionEvidence,
    Severity,
)


class InspectionRiskEngine:
    """Fuse independent inspection evidence into risk and severity."""

    def calculate(
        self,
        evidence: list[InspectionEvidence],
    ) -> tuple[float, InspectionDecision, Severity, float]:

        if not evidence:
            raise ValueError("at least one evidence item is required")

        total_weight = sum(item.weight for item in evidence)

        if total_weight <= 0:
            raise ValueError("total evidence weight must be greater than zero")

        weighted_risk = sum(
            item.weighted_score for item in evidence
        ) / total_weight

        abnormal_count = sum(item.abnormal for item in evidence)

        corroboration = abnormal_count / len(evidence)

        # Independent abnormal signals increase confidence in the risk.
        risk = float(
            np.clip(
                (0.80 * weighted_risk)
                + (0.20 * corroboration),
                0.0,
                1.0,
            )
        )

        decision = self._decision(risk)
        severity = self._severity(risk)
        confidence = self._confidence(
            risk=risk,
            corroboration=corroboration,
            decision=decision,
        )

        return risk, decision, severity, confidence

    @staticmethod
    def _decision(risk: float) -> InspectionDecision:
        if risk < 0.20:
            return InspectionDecision.PASS

        if risk < 0.45:
            return InspectionDecision.REVIEW

        return InspectionDecision.FAIL

    @staticmethod
    def _severity(risk: float) -> Severity:
        if risk < 0.10:
            return Severity.NONE

        if risk < 0.25:
            return Severity.LOW

        if risk < 0.45:
            return Severity.MEDIUM

        if risk < 0.70:
            return Severity.HIGH

        return Severity.CRITICAL

    @staticmethod
    def _confidence(
        *,
        risk: float,
        corroboration: float,
        decision: InspectionDecision,
    ) -> float:

        if decision is InspectionDecision.PASS:
            decision_margin = 1.0 - min(risk / 0.20, 1.0)

        elif decision is InspectionDecision.FAIL:
            decision_margin = min(
                max((risk - 0.45) / 0.55, 0.0),
                1.0,
            )

        else:
            decision_margin = 1.0 - min(
                abs(risk - 0.325) / 0.125,
                1.0,
            )

        confidence = (
            0.60 * decision_margin
            + 0.40 * (
                corroboration
                if decision is not InspectionDecision.PASS
                else 1.0 - corroboration
            )
        )

        return float(np.clip(confidence, 0.0, 1.0))