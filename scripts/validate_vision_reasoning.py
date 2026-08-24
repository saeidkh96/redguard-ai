from __future__ import annotations

from redguard.intelligence import (
    InspectionIntelligenceEngine,
)
from redguard.reasoning import (
    RuleBasedReasoner,
    VisionReasoningEngine,
)


def main() -> None:
    print(
        "RedGuard AI Vision Reasoning Validation"
    )
    print("=" * 42)

    intelligence = (
        InspectionIntelligenceEngine()
    )

    reasoning = VisionReasoningEngine(
        RuleBasedReasoner()
    )

    normal = intelligence.inspect(
        component_id="R27",
        component_type="resistor",
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.04,
        edge_difference=0.01,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    altered = intelligence.inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.75,
        texture_difference=0.70,
        reference_consensus=0.0,
    )

    normal_reasoned = reasoning.reason(
        normal
    )

    altered_reasoned = reasoning.reason(
        altered
    )

    print()
    print("NORMAL COMPONENT")
    print("----------------")

    print(
        f"Decision:    "
        f"{normal_reasoned.finding.decision.value}"
    )

    print(
        f"Severity:    "
        f"{normal_reasoned.finding.severity.value}"
    )

    print(
        f"Risk:        "
        f"{normal_reasoned.finding.risk_score:.4f}"
    )

    print(
        f"Explanation: "
        f"{normal_reasoned.reasoning.explanation}"
    )

    print()
    print("ALTERED COMPONENT")
    print("-----------------")

    print(
        f"Decision:    "
        f"{altered_reasoned.finding.decision.value}"
    )

    print(
        f"Severity:    "
        f"{altered_reasoned.finding.severity.value}"
    )

    print(
        f"Risk:        "
        f"{altered_reasoned.finding.risk_score:.4f}"
    )

    print("Observations:")

    for observation in (
        altered_reasoned
        .reasoning
        .observations
    ):
        print(
            f"  - {observation}"
        )

    print(
        f"Explanation: "
        f"{altered_reasoned.reasoning.explanation}"
    )

    print()

    checks = [
        (
            normal_reasoned.finding.decision
            is normal.decision,
            "Reasoning preserved PASS decision",
        ),
        (
            altered_reasoned.finding.decision
            is altered.decision,
            "Reasoning preserved FAIL decision",
        ),
        (
            altered_reasoned.finding.risk_score
            == altered.risk_score,
            "Reasoning preserved deterministic risk",
        ),
        (
            altered_reasoned.finding.severity
            is altered.severity,
            "Reasoning preserved deterministic severity",
        ),
        (
            len(
                altered_reasoned
                .reasoning
                .observations
            )
            >= 4,
            "Reasoning used structured abnormal evidence",
        ),
        (
            bool(
                altered_reasoned
                .reasoning
                .explanation
            ),
            "Contextual explanation generated",
        ),
    ]

    print("Validation")
    print("----------")

    failed = False

    for passed, message in checks:
        state = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{state}] {message}"
        )

        failed = (
            failed
            or not passed
        )

    print()

    if failed:
        raise SystemExit(
            "REDGUARD v0.8.0 "
            "VISION REASONING: FAIL"
        )

    print(
        "REDGUARD v0.8.0 "
        "VISION REASONING: PASS"
    )


if __name__ == "__main__":
    main()