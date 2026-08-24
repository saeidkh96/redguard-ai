from __future__ import annotations

import json

from redguard.intelligence import (
    InspectionDecision,
    InspectionIntelligenceEngine,
    InspectionReportBuilder,
)


def print_finding(finding) -> None:
    print(
        f"{finding.component_id:<8} "
        f"{finding.component_type:<20} "
        f"{finding.decision.value:<7} "
        f"severity={finding.severity.value:<8} "
        f"risk={finding.risk_score:.4f} "
        f"confidence={finding.confidence:.4f}"
    )

    for evidence in finding.abnormal_evidence:
        print(
            f"  - {evidence.evidence_type.value:<22} "
            f"score={evidence.score:.4f} "
            f"{evidence.description}"
        )

    print(f"  Explanation: {finding.explanation}")
    print()


def main() -> None:
    print("RedGuard AI Inspection Intelligence Validation")
    print("=" * 48)

    engine = InspectionIntelligenceEngine()

    normal = engine.inspect(
        component_id="R27",
        component_type="resistor",
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.04,
        edge_difference=0.01,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    altered = engine.inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.75,
        texture_difference=0.70,
        reference_consensus=0.0,
    )

    findings = [normal, altered]

    print()
    print("Component Findings")
    print("------------------")

    for finding in findings:
        print_finding(finding)

    report_builder = InspectionReportBuilder()
    report = report_builder.build(findings)
    payload = report_builder.to_dict(report)

    print("Report Summary")
    print("--------------")
    print(json.dumps(payload["summary"], indent=2))
    print()

    checks = [
        (
            normal.decision is InspectionDecision.PASS,
            "Known-normal component passed",
        ),
        (
            altered.decision is InspectionDecision.FAIL,
            "Altered component failed",
        ),
        (
            altered.risk_score > normal.risk_score,
            "Altered component has higher risk",
        ),
        (
            len(altered.abnormal_evidence) >= 4,
            "Multiple independent abnormal signals collected",
        ),
        (
            report.passed_count == 1 and report.failed_count == 1,
            "Structured report summarized inspection decisions",
        ),
        (
            bool(altered.explanation),
            "Explainable inspection finding generated",
        ),
    ]

    print("Validation")
    print("----------")

    failed = False

    for passed, message in checks:
        state = "PASS" if passed else "FAIL"
        print(f"[{state}] {message}")
        failed = failed or not passed

    print()

    if failed:
        raise SystemExit(
            "REDGUARD v0.7.0 INSPECTION INTELLIGENCE: FAIL"
        )

    print(
        "REDGUARD v0.7.0 INSPECTION INTELLIGENCE: PASS"
    )


if __name__ == "__main__":
    main()