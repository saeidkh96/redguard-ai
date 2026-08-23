import cv2
import numpy as np
import pytest

from redguard.detection.baseline import (
    BaselineComponentDetector,
)


def build_detection_image() -> np.ndarray:
    image = np.full(
        (400, 600, 3),
        30,
        dtype=np.uint8,
    )

    # Resistor
    cv2.rectangle(
        image,
        (60, 70),
        (180, 100),
        (220, 220, 220),
        -1,
    )

    # Capacitor
    cv2.rectangle(
        image,
        (250, 60),
        (285, 130),
        (220, 220, 220),
        -1,
    )

    # Transistor-like
    cv2.rectangle(
        image,
        (350, 70),
        (405, 115),
        (220, 220, 220),
        -1,
    )

    # IC
    cv2.rectangle(
        image,
        (180, 220),
        (290, 300),
        (220, 220, 220),
        -1,
    )

    return image


def test_detects_expected_component_count():
    image = build_detection_image()

    result = BaselineComponentDetector().detect(
        image
    )

    assert len(result) == 4


def test_detects_component_types():
    image = build_detection_image()

    result = BaselineComponentDetector().detect(
        image
    )

    types = {
        detection.component_type
        for detection in result
    }

    assert "resistor" in types
    assert "capacitor" in types
    assert "transistor" in types
    assert "integrated_circuit" in types


def test_detection_has_valid_bounding_boxes():
    image = build_detection_image()

    result = BaselineComponentDetector().detect(
        image
    )

    for detection in result:
        box = detection.bounding_box

        assert box.width > 0
        assert box.height > 0
        assert box.x >= 0
        assert box.y >= 0


def test_detection_confidence_is_valid():
    image = build_detection_image()

    result = BaselineComponentDetector().detect(
        image
    )

    for detection in result:
        assert 0.0 <= detection.confidence <= 1.0


def test_empty_scene_returns_no_detections():
    image = np.zeros(
        (300, 400, 3),
        dtype=np.uint8,
    )

    result = BaselineComponentDetector().detect(
        image
    )

    assert result == ()


def test_invalid_min_area_is_rejected():
    with pytest.raises(ValueError):
        BaselineComponentDetector(
            min_area=0
        )


def test_invalid_area_range_is_rejected():
    with pytest.raises(ValueError):
        BaselineComponentDetector(
            min_area=1000,
            max_area=500,
        )
