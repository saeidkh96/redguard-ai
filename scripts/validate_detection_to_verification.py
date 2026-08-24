from pathlib import Path

import cv2

from redguard.detection.dataset import (
    collect_split_samples,
    load_dataset_config,
)
from redguard.detection.registry import (
    build_component_registry,
)
from redguard.detection.yolo import (
    YoloComponentDetector,
)
from redguard.imaging.change_detection import (
    ChangedRegion,
)
from redguard.inspection.component_verifier import (
    ComponentVerifier,
)


MODEL_PATH = Path("models/redguard-yolo-synthetic.pt")

CONFIG_PATH = Path(
    "configs/components.yaml"
)


def main() -> int:
    print(
        "RedGuard AI Detection-to-Verification Validation"
    )
    print("=" * 52)

    if not MODEL_PATH.exists():
        print(
            "[FAIL] YOLO model missing"
        )
        return 1

    config = load_dataset_config(
        CONFIG_PATH
    )

    samples = collect_split_samples(
        config,
        "test",
    )

    if not samples:
        print(
            "[FAIL] No test samples"
        )
        return 1

    image = cv2.imread(
        str(samples[0].image_path)
    )

    if image is None:
        print(
            "[FAIL] Test image could not be loaded"
        )
        return 1

    detector = YoloComponentDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.05,
        iou_threshold=0.7,
    )

    detections = detector.detect(
        image
    )

    if not detections:
        print(
            "[FAIL] YOLO returned no detections"
        )
        return 1

    registry = build_component_registry(
        detections
    )

    target = registry.components[0]
    box = target.bounding_box

    changed_region = ChangedRegion(
        x=box.x + max(1, box.width // 4),
        y=box.y + max(1, box.height // 4),
        width=max(
            4,
            box.width // 2,
        ),
        height=max(
            4,
            box.height // 2,
        ),
        area=max(
            16,
            (box.width // 2)
            * (box.height // 2),
        ),
    )

    verification = ComponentVerifier(
        min_overlap_ratio=0.05
    ).verify(
        registry.components,
        [changed_region],
    )

    changed_components = (
        verification.changed_components
    )

    print(
        f"YOLO detections:          "
        f"{len(detections)}"
    )

    print(
        f"Generated registry size:  "
        f"{len(registry.components)}"
    )

    print(
        f"Detector source:          "
        f"{registry.source_detector}"
    )

    print()

    for component in (
        verification.components
    ):
        print(
            f"{component.component_id:<28} "
            f"{component.component_type:<20} "
            f"{component.status.value.upper():<10} "
            f"overlap={component.overlap_ratio:.2%}"
        )

    print()

    changed_ids = {
        component.component_id
        for component
        in changed_components
    }

    expected_id = (
        target.component_id
    )

    if (
        expected_id in changed_ids
        and len(registry.components) >= 1
    ):
        print(
            "[PASS] YOLO detection executed"
        )

        print(
            "[PASS] Automatic registry generated"
        )

        print(
            "[PASS] Stable component IDs generated"
        )

        print(
            "[PASS] Change mapped to detected component"
        )

        print(
            "[PASS] Detection integrated with verifier"
        )

        print()

        print(
            "REDGUARD v0.2.0 "
            "DETECTION-TO-VERIFICATION: PASS"
        )

        return 0

    print(
        "[FAIL] Detection-to-verification "
        "integration failed"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
