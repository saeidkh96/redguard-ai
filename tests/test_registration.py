import cv2
import numpy as np
import pytest

from redguard.core.exceptions import ImageValidationError
from redguard.imaging.registration import ImageRegistrationEngine


def build_feature_image() -> np.ndarray:
    image = np.full(
        (500, 700, 3),
        25,
        dtype=np.uint8,
    )

    # PCB body
    cv2.rectangle(
        image,
        (45, 45),
        (655, 455),
        (65, 115, 65),
        -1,
    )

    # Large IC
    cv2.rectangle(
        image,
        (250, 170),
        (450, 320),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        "RG-A17",
        (285, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    # Unique component shapes
    cv2.circle(
        image,
        (110, 110),
        22,
        (230, 230, 230),
        -1,
    )

    cv2.rectangle(
        image,
        (520, 90),
        (600, 125),
        (180, 180, 180),
        -1,
    )

    cv2.rectangle(
        image,
        (100, 360),
        (155, 420),
        (15, 15, 15),
        -1,
    )

    cv2.circle(
        image,
        (580, 390),
        28,
        (20, 20, 20),
        -1,
    )

    # PCB traces
    cv2.line(
        image,
        (120, 150),
        (250, 200),
        (190, 190, 190),
        3,
    )

    cv2.line(
        image,
        (450, 280),
        (570, 350),
        (190, 190, 190),
        3,
    )

    cv2.line(
        image,
        (180, 400),
        (300, 320),
        (190, 190, 190),
        3,
    )

    # Unique labels
    labels = [
        ("Q14", (85, 90)),
        ("R27", (510, 80)),
        ("C08", (90, 350)),
        ("U3", (555, 350)),
        ("J1", (470, 420)),
    ]

    for text, position in labels:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Mounting holes
    for point in (
        (75, 75),
        (625, 75),
        (75, 425),
        (625, 425),
    ):
        cv2.circle(
            image,
            point,
            10,
            (5, 5, 5),
            -1,
        )

    return image


def transform_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        3.0,
        1.0,
    )

    matrix[0, 2] += 12
    matrix[1, 2] -= 8

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
    )


def mean_absolute_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                first.astype(np.float32)
                - second.astype(np.float32)
            )
        )
    )


def test_registration_reduces_alignment_error():
    reference = build_feature_image()
    inspection = transform_image(reference)

    before = mean_absolute_difference(
        reference,
        inspection,
    )

    result = ImageRegistrationEngine().register(
        reference,
        inspection,
    )

    after = mean_absolute_difference(
        reference,
        result.aligned_image,
    )

    assert after < before
    assert result.matches_used >= 4
    assert result.inliers >= 4
    assert 0.0 <= result.inlier_ratio <= 1.0


def test_registered_image_matches_reference_dimensions():
    reference = build_feature_image()
    inspection = transform_image(reference)

    result = ImageRegistrationEngine().register(
        reference,
        inspection,
    )

    assert result.aligned_image.shape == reference.shape


def test_invalid_small_reference_is_rejected():
    reference = np.zeros((10, 10, 3), dtype=np.uint8)
    inspection = build_feature_image()

    with pytest.raises(ImageValidationError):
        ImageRegistrationEngine().register(
            reference,
            inspection,
        )


def test_featureless_images_are_rejected():
    reference = np.zeros((200, 200, 3), dtype=np.uint8)
    inspection = np.zeros((200, 200, 3), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        ImageRegistrationEngine().register(
            reference,
            inspection,
        )
