from pathlib import Path

import cv2
from ultralytics import YOLO

from redguard.detection.dataset import (
    collect_split_samples,
    load_dataset_config,
)
from redguard.detection.yolo import (
    YoloComponentDetector,
)


MODEL_PATH = Path(
    "runs/detect/artifacts/"
    "detection_training/"
    "v020_baseline/"
    "weights/"
    "best.pt"
)

CONFIG_PATH = Path(
    "configs/components.yaml"
)

ARTIFACT_DIR = Path(
    "artifacts/yolo_detection"
)


def draw_detections(
    image,
    detections,
):
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
    print(
        "RedGuard AI YOLO Detector Validation"
    )
    print("=" * 38)

    if not MODEL_PATH.exists():
        print(
            f"[FAIL] Trained model not found: "
            f"{MODEL_PATH}"
        )
        return 1

    if not CONFIG_PATH.exists():
        print(
            f"[FAIL] Dataset config not found: "
            f"{CONFIG_PATH}"
        )
        return 1

    config = load_dataset_config(
        CONFIG_PATH
    )

    test_samples = collect_split_samples(
        config,
        "test",
    )

    if not test_samples:
        print(
            "[FAIL] Test dataset is empty."
        )
        return 1

    #
    # 1. Proper object-detection evaluation
    #
    model = YOLO(
        str(MODEL_PATH)
    )

    metrics = model.val(
        data=str(CONFIG_PATH),
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project="artifacts/yolo_detection",
        name="test_metrics",
        exist_ok=True,
        verbose=False,
    )

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map50_95 = float(
        metrics.box.map
    )

    #
    # 2. Validate RedGuard detector abstraction
    #
    detector = YoloComponentDetector(
        MODEL_PATH,
        confidence_threshold=0.05,
        iou_threshold=0.7,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_detections = 0
    images_with_detections = 0

    class_counts = {
        class_name: 0
        for class_name
        in config.class_names.values()
    }

    for index, sample in enumerate(
        test_samples
    ):
        image = cv2.imread(
            str(sample.image_path)
        )

        if image is None:
            print(
                f"[FAIL] Unable to read image: "
                f"{sample.image_path}"
            )
            return 1

        detections = detector.detect(
            image
        )

        total_detections += len(
            detections
        )

        if detections:
            images_with_detections += 1

        for detection in detections:
            if (
                detection.component_type
                in class_counts
            ):
                class_counts[
                    detection.component_type
                ] += 1

        if index == 0:
            annotated = draw_detections(
                image,
                detections,
            )

            cv2.imwrite(
                str(
                    ARTIFACT_DIR
                    / "sample_detection.png"
                ),
                annotated,
            )

    print(
        f"Test images:              "
        f"{len(test_samples)}"
    )

    print(
        f"Images with detections:   "
        f"{images_with_detections}"
    )

    print(
        f"Model detections:         "
        f"{total_detections}"
    )

    print()

    print("Detection metrics")
    print("-----------------")

    print(
        f"Precision:                "
        f"{precision:.4f}"
    )

    print(
        f"Recall:                   "
        f"{recall:.4f}"
    )

    print(
        f"mAP50:                    "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95:                 "
        f"{map50_95:.4f}"
    )

    print()

    print("Detected classes")
    print("----------------")

    for class_name, count in (
        class_counts.items()
    ):
        print(
            f"{class_name:<22} "
            f"{count}"
        )

    print()

    all_images_detected = (
        images_with_detections
        == len(test_samples)
    )

    all_classes_detected = all(
        count > 0
        for count
        in class_counts.values()
    )

    metrics_pass = (
        precision >= 0.75
        and recall >= 0.75
        and map50 >= 0.85
        and map50_95 >= 0.70
    )

    if (
        all_images_detected
        and all_classes_detected
        and metrics_pass
    ):
        print(
            "[PASS] Trained YOLO model loaded"
        )

        print(
            "[PASS] RedGuard detector abstraction executed"
        )

        print(
            "[PASS] Every test image produced detections"
        )

        print(
            "[PASS] All component classes detected"
        )

        print(
            "[PASS] Precision threshold satisfied"
        )

        print(
            "[PASS] Recall threshold satisfied"
        )

        print(
            "[PASS] mAP50 threshold satisfied"
        )

        print(
            "[PASS] mAP50-95 threshold satisfied"
        )

        print(
            "[PASS] Detection artifacts generated"
        )

        print()

        print(
            "REDGUARD v0.2.0 "
            "YOLO DETECTOR: PASS"
        )

        return 0

    print(
        "[FAIL] YOLO detector "
        "validation failed"
    )

    if not all_images_detected:
        print(
            "[FAIL] Some test images "
            "had no detections"
        )

    if not all_classes_detected:
        print(
            "[FAIL] Not all component "
            "classes were detected"
        )

    if not metrics_pass:
        print(
            "[FAIL] Detection metrics "
            "below release thresholds"
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
