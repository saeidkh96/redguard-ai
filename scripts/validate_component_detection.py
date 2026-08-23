from pathlib import Path

import cv2
import numpy as np

from redguard.detection.baseline import (
    BaselineComponentDetector,
)


ARTIFACT_DIR = Path(
    "artifacts/component_detection"
)


def build_scene() -> np.ndarray:
    image = np.full(
        (500, 800, 3),
        30,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (70, 90),
        (210, 125),
        (225, 225, 225),
        -1,
    )

    cv2.rectangle(
        image,
        (300, 75),
        (340, 155),
        (225, 225, 225),
        -1,
    )

    cv2.rectangle(
        image,
        (430, 90),
        (490, 140),
        (225, 225, 225),
        -1,
    )

    cv2.rectangle(
        image,
        (280, 280),
        (420, 380),
        (225, 225, 225),
        -1,
    )

    return image


def draw_detections(
    image: np.ndarray,
    detections,
) -> np.ndarray:
    output = image.copy()

    for detection in detections:
        box = detection.bounding_box

        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection.component_type} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                box.x,
                max(20, box.y - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def main() -> int:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene = build_scene()

    detector = BaselineComponentDetector()

    detections = detector.detect(
        scene
    )

    annotated = draw_detections(
        scene,
        detections,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "detection_input.png"
        ),
        scene,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "detection_output.png"
        ),
        annotated,
    )

    print(
        "RedGuard AI Automatic Component "
        "Detection Validation"
    )
    print("=" * 52)

    print(
        f"Detected components: "
        f"{len(detections)}"
    )

    print()

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        box = detection.bounding_box

        print(
            f"#{index} "
            f"{detection.component_type:<20} "
            f"confidence={detection.confidence:.3f} "
            f"bbox=({box.x},{box.y},"
            f"{box.width},{box.height})"
        )

    print()

    detected_types = {
        detection.component_type
        for detection in detections
    }

    expected_types = {
        "resistor",
        "capacitor",
        "transistor",
        "integrated_circuit",
    }

    if (
        len(detections) == 4
        and detected_types == expected_types
    ):
        print(
            "[PASS] Components detected automatically"
        )
        print(
            "[PASS] Component types classified"
        )
        print(
            "[PASS] Bounding boxes generated"
        )
        print(
            "[PASS] Detection confidence generated"
        )
        print(
            "[PASS] Detection artifacts generated"
        )
        print()
        print(
            "REDGUARD v0.2.0 "
            "BASELINE DETECTION: PASS"
        )
        return 0

    print(
        "[FAIL] Automatic component "
        "detection validation failed"
    )

    print()

    print(
        "REDGUARD v0.2.0 "
        "BASELINE DETECTION: FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
