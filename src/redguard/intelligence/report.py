from __future__ import annotations

from redguard.intelligence.models import (
    InspectionFinding,
    InspectionReport,
)


class InspectionReportBuilder:
    """Build structured reports from component inspection findings."""

    def build(
        self,
        findings: list[InspectionFinding],
    ) -> InspectionReport:
        return InspectionReport(findings=list(findings))

    @staticmethod
    def to_dict(report: InspectionReport) -> dict[str, object]:
        return {
            "summary": {
                "components": report.component_count,
                "passed": report.passed_count,
                "review": report.review_count,
                "failed": report.failed_count,
                "maximum_risk": round(report.maximum_risk, 6),
            },
            "findings": [
                {
                    "component_id": finding.component_id,
                    "component_type": finding.component_type,
                    "decision": finding.decision.value,
                    "severity": finding.severity.value,
                    "risk_score": round(finding.risk_score, 6),
                    "confidence": round(finding.confidence, 6),
                    "explanation": finding.explanation,
                    "evidence": [
                        {
                            "type": evidence.evidence_type.value,
                            "score": round(evidence.score, 6),
                            "weight": evidence.weight,
                            "abnormal": evidence.abnormal,
                            "description": evidence.description,
                        }
                        for evidence in finding.evidence
                    ],
                }
                for finding in report.findings
            ],
        }