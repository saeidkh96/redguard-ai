import cv2
import numpy as np
import pytest

from redguard.imaging.change_detection import ChangeDetector


def build_reference() -> np.ndarray:
    image = np.full(
        (300, 400, 3),
        40,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (50, 50),
        (350, 250),
        (80, 120, 80),
        -1,
    )

    cv2.rectangle(
        image,
        (150, 110),
        (250, 190),
        (25, 25, 25),
        -1,
    )

    cv2.putText(
        image,
        "Q14",
        (175, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    return image


def test_identical_images_have_no_change():
    reference = build_reference()

    result = ChangeDetector().detect(
        reference,
        reference.copy(),
    )

    assert result.changed is False
    assert result.similarity == pytest.approx(1.0)
    assert result.changed_area == 0
    assert result.changed_area_ratio == 0.0
    assert len(result.regions) == 0


def test_detects_local_component_change():
    reference = build_reference()
    inspection = reference.copy()

    cv2.rectangle(
        inspection,
        (280, 170),
        (320, 210),
        (240, 240, 240),
        -1,
    )

    result = ChangeDetector(
        threshold=30,
        min_region_area=50,
    ).detect(
        reference,
        inspection,
    )

    assert result.changed is True
    assert result.similarity < 1.0
    assert result.changed_area > 0
    assert result.changed_area_ratio > 0.0
    assert len(result.regions) >= 1


def test_small_noise_is_filtered():
    reference = build_reference()
    inspection = reference.copy()

    inspection[100, 100] = (255, 255, 255)

    result = ChangeDetector(
        min_region_area=40,
    ).detect(
        reference,
        inspection,
    )

    assert result.changed is False


def test_dimension_mismatch_is_rejected():
    reference = build_reference()

    inspection = np.zeros(
        (350, 400, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        ChangeDetector().detect(
            reference,
            inspection,
        )


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        ChangeDetector(threshold=300)


def test_regions_include_location():
    reference = build_reference()
    inspection = reference.copy()

    cv2.rectangle(
        inspection,
        (275, 165),
        (325, 215),
        (255, 255, 255),
        -1,
    )

    result = ChangeDetector(
        threshold=30,
        min_region_area=50,
    ).detect(
        reference,
        inspection,
    )

    region = result.regions[0]

    assert region.x >= 0
    assert region.y >= 0
    assert region.width > 0
    assert region.height > 0
    assert region.area > 0
