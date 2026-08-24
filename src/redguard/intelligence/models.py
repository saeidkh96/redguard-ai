from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EvidenceType(StrEnum):
    VISUAL_CHANGE = "visual_change"
    FINGERPRINT = "fingerprint"
    ANOMALY = "anomaly"
    EDGE = "edge"
    TEXTURE = "texture"
    REFERENCE_CONSENSUS = "reference_consensus"


class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InspectionDecision(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass(slots=True, frozen=True)
class InspectionEvidence:
    evidence_type: EvidenceType
    score: float
    weight: float
    description: str
    abnormal: bool

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass(slots=True)
class InspectionFinding:
    component_id: str
    component_type: str
    decision: InspectionDecision
    severity: Severity
    risk_score: float
    confidence: float
    evidence: list[InspectionEvidence] = field(default_factory=list)
    explanation: str = ""

    @property
    def abnormal_evidence(self) -> list[InspectionEvidence]:
        return [item for item in self.evidence if item.abnormal]


@dataclass(slots=True)
class InspectionReport:
    findings: list[InspectionFinding]

    @property
    def component_count(self) -> int:
        return len(self.findings)

    @property
    def failed_count(self) -> int:
        return sum(
            finding.decision is InspectionDecision.FAIL
            for finding in self.findings
        )

    @property
    def review_count(self) -> int:
        return sum(
            finding.decision is InspectionDecision.REVIEW
            for finding in self.findings
        )

    @property
    def passed_count(self) -> int:
        return sum(
            finding.decision is InspectionDecision.PASS
            for finding in self.findings
        )

    @property
    def maximum_risk(self) -> float:
        if not self.findings:
            return 0.0
        return max(finding.risk_score for finding in self.findings)