from __future__ import annotations

import numpy as np

from redguard.inspection.multi_reference import (
    MultiReferenceVerifier,
    VerificationDecision,
)
from redguard.reference import ReferenceSample, ReferenceSet


def normalized(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def build_reference_set() -> ReferenceSet:
    reference_set = ReferenceSet(
        component_id="Q14",
        component_type="transistor",
    )

    references = [
        ("normal_light", [1.00, 0.02, 0.01, 0.00]),
        ("normal_dark", [0.99, 0.04, 0.01, 0.01]),
        ("normal_view", [1.00, -0.02, 0.03, 0.01]),
    ]

    for sample_id, values in references:
        reference_set.add(
            ReferenceSample(
                sample_id=sample_id,
                component_id="Q14",
                component_type="transistor",
                fingerprint=normalized(values),
            )
        )

    return reference_set


def print_result(name: str, result) -> None:
    print(name)
    print("-" * len(name))

    for match in result.match.matches:
        state = "ACCEPT" if match.accepted else "REJECT"
        print(
            f"{match.sample_id:<16} "
            f"similarity={match.similarity:.4f} "
            f"{state}"
        )

    print()
    print(f"Best reference:    {result.match.best_sample_id}")
    print(f"Best similarity:   {result.match.best_similarity:.4f}")
    print(f"Mean similarity:   {result.match.mean_similarity:.4f}")
    print(f"Consensus ratio:   {result.match.consensus_ratio:.3f}")
    print(f"Risk score:        {result.risk_score:.4f}")
    print(f"Confidence:        {result.confidence:.4f}")
    print(f"Decision:          {result.decision.value}")
    print()


def main() -> None:
    print("RedGuard AI Multi-Reference Verification")
    print("=" * 44)

    reference_set = build_reference_set()

    verifier = MultiReferenceVerifier(
        similarity_threshold=0.97,
        pass_consensus=0.67,
        review_consensus=0.34,
    )

    normal = verifier.verify(
        normalized([0.995, 0.025, 0.015, 0.005]),
        reference_set,
    )

    replacement = verifier.verify(
        normalized([0.60, 0.65, 0.40, 0.20]),
        reference_set,
    )

    print(f"Reference population: {reference_set.size}")
    print()

    print_result("NORMAL INSPECTION", normal)
    print_result("REPLACEMENT INSPECTION", replacement)

    checks = [
        (
            normal.decision is VerificationDecision.PASS,
            "Normal component accepted by reference population",
        ),
        (
            normal.match.consensus_ratio >= 0.67,
            "Normal component reached reference consensus",
        ),
        (
            replacement.decision is VerificationDecision.FAIL,
            "Replacement-like component rejected",
        ),
        (
            replacement.match.consensus_ratio == 0.0,
            "Replacement-like component failed reference consensus",
        ),
        (
            replacement.risk_score > normal.risk_score,
            "Replacement risk exceeds normal risk",
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
            "REDGUARD v0.6.0 MULTI-REFERENCE VERIFICATION: FAIL"
        )

    print(
        "REDGUARD v0.6.0 MULTI-REFERENCE VERIFICATION: PASS"
    )


if __name__ == "__main__":
    main()