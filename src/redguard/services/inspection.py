from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from redguard.intelligence import InspectionIntelligenceEngine
from redguard.reasoning import RuleBasedReasoner, VisionReasoningEngine
from redguard.persistence import InspectionRecord, InspectionRepository
from .artifacts import ArtifactService


class InspectionService:
    """Application service: deterministic intelligence -> safe reasoning -> persistence."""

    def __init__(self, repository: InspectionRepository | None = None, artifacts: ArtifactService | None = None) -> None:
        self.repository = repository or InspectionRepository()
        self.artifacts = artifacts or ArtifactService()
        self.intelligence = InspectionIntelligenceEngine()
        self.reasoning = VisionReasoningEngine(RuleBasedReasoner())

    def inspect(self, *, component_id: str, component_type: str, change_score: float,
                fingerprint_similarity: float, anomaly_score: float, edge_difference: float,
                texture_difference: float, reference_consensus: float) -> InspectionRecord:
        finding = self.intelligence.inspect(
            component_id=component_id, component_type=component_type,
            change_score=change_score, fingerprint_similarity=fingerprint_similarity,
            anomaly_score=anomaly_score, edge_difference=edge_difference,
            texture_difference=texture_difference, reference_consensus=reference_consensus,
        )
        reasoned = self.reasoning.reason(finding)
        inspection_id = uuid4().hex
        evidence = [
            {
                "type": item.evidence_type.value,
                "score": float(item.score),
                "weight": float(item.weight),
                "abnormal": bool(item.abnormal),
                "description": item.description,
            }
            for item in finding.evidence
        ]
        record = InspectionRecord(
            inspection_id=inspection_id,
            component_id=finding.component_id,
            component_type=finding.component_type,
            decision=finding.decision.value,
            severity=finding.severity.value,
            risk_score=float(finding.risk_score),
            confidence=float(finding.confidence),
            explanation=reasoned.reasoning.explanation,
            evidence=evidence,
        )
        report_path = self.artifacts.write_json(inspection_id, "report.json", record.to_dict())
        record.artifacts.append(report_path)
        self.repository.save(record)
        return record

    def get(self, inspection_id: str) -> InspectionRecord | None:
        return self.repository.get(inspection_id)

    def history(self) -> list[InspectionRecord]:
        return self.repository.list()
