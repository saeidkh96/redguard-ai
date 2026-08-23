import cv2
import numpy as np

from redguard.features.backbone import VisionBackbone
from redguard.features.fingerprint import ComponentFingerprinter


def build_reference():
    image = np.full(
        (224, 224, 3),
        35,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (55, 65),
        (170, 160),
        (175, 175, 175),
        -1,
    )

    cv2.putText(
        image,
        "Q14 A7",
        (70, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
    )

    cv2.line(
        image,
        (60, 145),
        (160, 145),
        (90, 90, 90),
        2,
    )

    return image


def main():
    reference = build_reference()

    same = cv2.convertScaleAbs(
        reference,
        alpha=0.97,
        beta=3,
    )

    replacement = reference.copy()

    cv2.rectangle(
        replacement,
        (80, 82),
        (150, 145),
        (65, 65, 65),
        -1,
    )

    cv2.putText(
        replacement,
        "Q14 A7",
        (70, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        2,
    )

    fingerprinter = ComponentFingerprinter(
        backbone=VisionBackbone(
            pretrained=True
        ),
        similarity_threshold=0.97,
    )

    same_result = fingerprinter.compare(
        reference,
        same,
    )

    replacement_result = fingerprinter.compare(
        reference,
        replacement,
    )

    print(
        "RedGuard AI Visual Fingerprint Validation"
    )
    print("=" * 42)

    print(
        f"Decision threshold:             "
        f"{fingerprinter.similarity_threshold:.4f}"
    )

    print(
        f"Same-instance-like similarity:  "
        f"{same_result.similarity:.4f}"
    )

    print(
        f"Same-instance decision:         "
        f"{same_result.same_instance_candidate}"
    )

    print(
        f"Replacement similarity:         "
        f"{replacement_result.similarity:.4f}"
    )

    print(
        f"Replacement same-instance:      "
        f"{replacement_result.same_instance_candidate}"
    )

    print()

    if (
        same_result.same_instance_candidate
        and not replacement_result.same_instance_candidate
        and same_result.similarity
        > replacement_result.similarity
    ):
        print(
            "[PASS] Stable visual fingerprint generated"
        )
        print(
            "[PASS] Same-instance-like sample accepted"
        )
        print(
            "[PASS] Replacement-like sample rejected"
        )
        print(
            "[PASS] Identity threshold discriminates samples"
        )
        print()

        print(
            "REDGUARD v0.3.0 "
            "VISUAL FINGERPRINTING: PASS"
        )

        return 0

    print(
        "[FAIL] Fingerprint identity "
        "validation failed"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
