from pathlib import Path

import cv2
import numpy as np
import pytest

from redguard.detection.yolo import (
    YoloComponentDetector,
)


MODEL_PATH = Path("models/redguard-yolo-synthetic.pt")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="trained YOLO model not available",
)
def test_yolo_detector_returns_components():
    image = np.full(
        (480, 640, 3),
        28,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (70, 80),
        (190, 110),
        (220, 220, 220),
        -1,
    )

    cv2.rectangle(
        image,
        (300, 75),
        (338, 150),
        (220, 220, 220),
        -1,
    )

    cv2.rectangle(
        image,
        (90, 290),
        (150, 338),
        (220, 220, 220),
        -1,
    )

    cv2.rectangle(
        image,
        (350, 275),
        (465, 360),
        (220, 220, 220),
        -1,
    )

    detector = YoloComponentDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.10,
    )

    detections = detector.detect(
        image
    )

    assert len(detections) >= 1

    for detection in detections:
        assert detection.component_type in {
            "transistor",
            "resistor",
            "capacitor",
            "integrated_circuit",
        }

        assert 0.0 <= detection.confidence <= 1.0

        assert detection.bounding_box.width > 0
        assert detection.bounding_box.height > 0


def test_missing_yolo_model_is_rejected():
    with pytest.raises(FileNotFoundError):
        YoloComponentDetector(
            model_path="missing-model.pt"
        )


def test_invalid_confidence_threshold():
    if not MODEL_PATH.exists():
        pytest.skip(
            "trained YOLO model not available"
        )

    with pytest.raises(ValueError):
        YoloComponentDetector(
            model_path=MODEL_PATH,
            confidence_threshold=2.0,
        )
