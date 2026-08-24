from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ComponentSignals:
    component_id: str
    component_type: str
    change_score: float
    fingerprint_similarity: float
    anomaly_score: float
    edge_difference: float
    texture_difference: float
    reference_consensus: float

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must not be empty")
        if not self.component_type:
            raise ValueError("component_type must not be empty")

        for name in (
            "change_score",
            "fingerprint_similarity",
            "anomaly_score",
            "edge_difference",
            "texture_difference",
            "reference_consensus",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_inspection_kwargs(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "change_score": self.change_score,
            "fingerprint_similarity": self.fingerprint_similarity,
            "anomaly_score": self.anomaly_score,
            "edge_difference": self.edge_difference,
            "texture_difference": self.texture_difference,
            "reference_consensus": self.reference_consensus,
        }


@dataclass(slots=True, frozen=True)
class PipelineDiagnostics:
    registration_inlier_ratio: float
    change_area_ratio: float
    changed_region_count: int
    detection_count: int
    component_count: int


@dataclass(slots=True)
class PipelineRunResult:
    diagnostics: PipelineDiagnostics
    quality_passed: bool
    quality_messages: tuple[str, ...]
    inspection_ids: list[str] = field(default_factory=list)
    decisions: dict[str, str] = field(default_factory=dict)
