from __future__ import annotations

from dataclasses import dataclass

from redguard.orchestration.models import PipelineDiagnostics


@dataclass(slots=True, frozen=True)
class QualityGateResult:
    passed: bool
    messages: tuple[str, ...]


class PipelineQualityGate:
    """Fail closed when core visual prerequisites are unreliable."""

    def __init__(
        self,
        *,
        min_registration_inlier_ratio: float = 0.30,
        min_detection_count: int = 1,
        max_change_area_ratio: float = 0.50,
    ) -> None:
        self.min_registration_inlier_ratio = min_registration_inlier_ratio
        self.min_detection_count = min_detection_count
        self.max_change_area_ratio = max_change_area_ratio

    def evaluate(self, diagnostics: PipelineDiagnostics) -> QualityGateResult:
        messages: list[str] = []

        if diagnostics.registration_inlier_ratio < self.min_registration_inlier_ratio:
            messages.append("registration quality below threshold")

        if diagnostics.detection_count < self.min_detection_count:
            messages.append("no reliable components detected")

        if diagnostics.component_count < self.min_detection_count:
            messages.append("component registry is empty")

        if diagnostics.change_area_ratio > self.max_change_area_ratio:
            messages.append("changed area exceeds quality limit")

        return QualityGateResult(
            passed=not messages,
            messages=tuple(messages),
        )
