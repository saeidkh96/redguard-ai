import cv2
import numpy as np
import pytest

from redguard.anomaly.patch_memory import (
    PatchMemoryAnomalyDetector,
)
from redguard.features.backbone import (
    VisionBackbone,
)


@pytest.fixture(scope="module")
def backbone():
    return VisionBackbone(
        pretrained=False
    )


def normal_component():
    image = np.full(
        (160, 160, 3),
        30,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (40, 50),
        (120, 110),
        (180, 180, 180),
        -1,
    )

    return image


def anomalous_component():
    image = normal_component()

    cv2.circle(
        image,
        (80, 80),
        22,
        (0, 0, 0),
        -1,
    )

    return image


def test_requires_fit(backbone):
    detector = PatchMemoryAnomalyDetector(
        backbone=backbone
    )

    with pytest.raises(RuntimeError):
        detector.predict(
            normal_component()
        )


def test_fit_builds_memory_bank(backbone):
    detector = PatchMemoryAnomalyDetector(
        backbone=backbone
    )

    detector.fit(
        [normal_component()]
    )

    assert detector.fitted
    assert len(detector.memory_bank) > 0


def test_result_has_anomaly_map(backbone):
    detector = PatchMemoryAnomalyDetector(
        backbone=backbone
    )

    image = normal_component()

    detector.fit([image])

    result = detector.predict(
        image
    )

    assert result.anomaly_map.shape == image.shape[:2]
    assert result.score >= 0.0


def test_anomaly_differs_from_reference(backbone):
    detector = PatchMemoryAnomalyDetector(
        backbone=backbone
    )

    normal = normal_component()

    detector.fit([normal])

    normal_result = detector.predict(
        normal
    )

    anomaly_result = detector.predict(
        anomalous_component()
    )

    assert (
        anomaly_result.score
        >= normal_result.score
    )
