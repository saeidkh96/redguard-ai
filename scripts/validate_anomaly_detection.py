from pathlib import Path

import cv2
import numpy as np

from redguard.anomaly.patch_memory import (
    PatchMemoryAnomalyDetector,
)
from redguard.features.backbone import VisionBackbone


ARTIFACT_DIR = Path(
    "artifacts/anomaly_detection"
)


def build_normal(seed=0):
    rng = np.random.default_rng(
        seed
    )

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

    noise = rng.integers(
        0,
        4,
        image.shape,
        dtype=np.uint8,
    )

    return cv2.add(
        image,
        noise,
    )


def main():
    normal_images = [
        build_normal(i)
        for i in range(6)
    ]

    normal_test = build_normal(
        20
    )

    anomalous = build_normal(
        30
    )

    cv2.circle(
        anomalous,
        (112, 112),
        28,
        (0, 0, 0),
        -1,
    )

    detector = PatchMemoryAnomalyDetector(
        backbone=VisionBackbone(
            pretrained=True
        ),
        anomaly_threshold=0.25,
    )

    detector.fit(
        normal_images
    )

    normal_result = detector.predict(
        normal_test
    )

    anomaly_result = detector.predict(
        anomalous
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized = cv2.normalize(
        anomaly_result.anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "anomaly_map.png"
        ),
        normalized,
    )

    print(
        "RedGuard AI Patch Anomaly Validation"
    )
    print("=" * 38)

    print(
        f"Decision threshold: "
        f"{detector.anomaly_threshold:.4f}"
    )

    print(
        f"Normal score:       "
        f"{normal_result.score:.4f}"
    )

    print(
        f"Normal anomalous:   "
        f"{normal_result.is_anomalous}"
    )

    print(
        f"Anomaly score:      "
        f"{anomaly_result.score:.4f}"
    )

    print(
        f"Anomaly detected:   "
        f"{anomaly_result.is_anomalous}"
    )

    print()

    if (
        not normal_result.is_anomalous
        and anomaly_result.is_anomalous
        and anomaly_result.score
        > normal_result.score
    ):
        print(
            "[PASS] Normal feature memory built"
        )
        print(
            "[PASS] Normal component accepted"
        )
        print(
            "[PASS] Anomalous component rejected"
        )
        print(
            "[PASS] Patch anomaly score discriminates samples"
        )
        print(
            "[PASS] Anomaly localization map generated"
        )
        print()

        print(
            "REDGUARD v0.4.0 "
            "ANOMALY DETECTION: PASS"
        )

        return 0

    print(
        "[FAIL] Anomaly decision "
        "validation failed"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
