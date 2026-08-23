import cv2
import numpy as np

from redguard.features.backbone import (
    VisionBackbone,
)
from redguard.features.fingerprint import (
    ComponentFingerprinter,
)
from redguard.inspection.fine_grained import (
    FineGrainedInspector,
    InspectionDecision,
)


def build_component():
    image = np.full(
        (160, 160, 3),
        30,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (35, 45),
        (125, 115),
        (180, 180, 180),
        -1,
    )

    cv2.putText(
        image,
        "Q14",
        (53, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
    )

    return image


def test_identical_component_passes():
    backbone = VisionBackbone(
        pretrained=False
    )

    inspector = FineGrainedInspector(
        fingerprinter=ComponentFingerprinter(
            backbone=backbone
        )
    )

    image = build_component()

    result = inspector.inspect(
        image,
        image.copy(),
    )

    assert (
        result.decision
        == InspectionDecision.PASS
    )

    assert result.risk_score < 0.25


def test_visual_change_increases_risk():
    backbone = VisionBackbone(
        pretrained=False
    )

    inspector = FineGrainedInspector(
        fingerprinter=ComponentFingerprinter(
            backbone=backbone
        )
    )

    reference = build_component()
    changed = reference.copy()

    cv2.line(
        changed,
        (30, 30),
        (130, 130),
        (255, 255, 255),
        8,
    )

    normal = inspector.inspect(
        reference,
        reference.copy(),
    )

    altered = inspector.inspect(
        reference,
        changed,
    )

    assert (
        altered.risk_score
        > normal.risk_score
    )
