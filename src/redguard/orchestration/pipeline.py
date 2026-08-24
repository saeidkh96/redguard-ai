from __future__ import annotations

import numpy as np

from redguard.orchestration.models import PipelineRunResult
from redguard.orchestration.quality import PipelineQualityGate
from redguard.orchestration.signals import ImageSignalExtractor, SignalExtractor
from redguard.services import InspectionService


class ProductionInspectionPipeline:
    """End-to-end orchestration from images to persisted inspection records."""

    def __init__(
        self,
        *,
        service: InspectionService | None = None,
        signal_extractor: SignalExtractor | None = None,
        quality_gate: PipelineQualityGate | None = None,
    ) -> None:
        self.service = service or InspectionService()
        self.signal_extractor = signal_extractor or ImageSignalExtractor()
        self.quality_gate = quality_gate or PipelineQualityGate()

    def run(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> PipelineRunResult:
        signals, diagnostics = self.signal_extractor.extract(
            reference,
            inspection,
        )

        quality = self.quality_gate.evaluate(diagnostics)

        if not quality.passed:
            return PipelineRunResult(
                diagnostics=diagnostics,
                quality_passed=False,
                quality_messages=quality.messages,
            )

        inspection_ids: list[str] = []
        decisions: dict[str, str] = {}

        for component_signals in signals:
            record = self.service.inspect(
                **component_signals.to_inspection_kwargs()
            )
            inspection_ids.append(record.inspection_id)
            decisions[record.component_id] = record.decision

        return PipelineRunResult(
            diagnostics=diagnostics,
            quality_passed=True,
            quality_messages=quality.messages,
            inspection_ids=inspection_ids,
            decisions=decisions,
        )
