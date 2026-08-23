from pathlib import Path

import cv2
import numpy as np

from redguard.imaging.change_detection import ChangeDetector
from redguard.imaging.registration import ImageRegistrationEngine


ARTIFACT_DIR = Path("artifacts/change_detection")


def build_reference() -> np.ndarray:
    image = np.full(
        (600, 900, 3),
        25,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (60, 60),
        (840, 540),
        (65, 115, 65),
        -1,
    )

    cv2.rectangle(
        image,
        (330, 200),
        (570, 390),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        "RG-CPU-A1",
        (370, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    components = [
        ((130, 130), (190, 165), "Q14"),
        ((670, 120), (760, 160), "R27"),
        ((120, 430), (185, 490), "C08"),
        ((680, 420), (755, 485), "U03"),
    ]

    for top_left, bottom_right, label in components:
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            (180, 180, 180),
            -1,
        )

        cv2.putText(
            image,
            label,
            (
                top_left[0],
                top_left[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    traces = [
        ((190, 150), (330, 230)),
        ((570, 250), (680, 150)),
        ((180, 450), (340, 350)),
        ((570, 350), (690, 450)),
    ]

    for start, end in traces:
        cv2.line(
            image,
            start,
            end,
            (210, 210, 210),
            3,
        )

    for point in (
        (95, 95),
        (805, 95),
        (95, 505),
        (805, 505),
    ):
        cv2.circle(
            image,
            point,
            14,
            (5, 5, 5),
            -1,
        )

    return image


def create_changed_board(
    reference: np.ndarray,
) -> np.ndarray:
    changed = reference.copy()

    cv2.rectangle(
        changed,
        (138, 137),
        (182, 158),
        (35, 35, 35),
        -1,
    )

    cv2.line(
        changed,
        (143, 142),
        (177, 153),
        (245, 245, 245),
        2,
    )

    return changed


def transform(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        3.5,
        1.0,
    )

    matrix[0, 2] += 15
    matrix[1, 2] -= 10

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
    )


def draw_regions(
    image: np.ndarray,
    regions,
) -> np.ndarray:
    output = image.copy()

    for region in regions:
        cv2.rectangle(
            output,
            (region.x, region.y),
            (
                region.x + region.width,
                region.y + region.height,
            ),
            (0, 0, 255),
            2,
        )

    return output


def main() -> int:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = build_reference()
    changed_board = create_changed_board(
        reference
    )
    inspection = transform(
        changed_board
    )

    registration_result = (
        ImageRegistrationEngine().register(
            reference,
            inspection,
        )
    )

    detector = ChangeDetector(
        threshold=25,
        ssim_threshold=20,
        min_region_area=80,
        morphology_kernel_size=3,
        comparison_blur_size=5,
        border_margin=12,
    )

    result = detector.detect(
        reference,
        registration_result.aligned_image,
        valid_mask=registration_result.valid_mask,
    )

    annotated = draw_regions(
        registration_result.aligned_image,
        result.regions,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "reference.png"
        ),
        reference,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "inspection_changed_transformed.png"
        ),
        inspection,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "inspection_aligned.png"
        ),
        registration_result.aligned_image,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "valid_registration_mask.png"
        ),
        registration_result.valid_mask,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "difference_map.png"
        ),
        result.difference_map,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "change_mask.png"
        ),
        result.binary_mask,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "detected_regions.png"
        ),
        annotated,
    )

    print(
        "RedGuard AI End-to-End "
        "Change Validation"
    )
    print("=" * 42)

    print(
        f"Registration inlier ratio: "
        f"{registration_result.inlier_ratio:.3f}"
    )

    print(
        f"Global similarity:          "
        f"{result.similarity:.4f}"
    )

    print(
        f"Changed:                    "
        f"{result.changed}"
    )

    print(
        f"Changed regions:            "
        f"{len(result.regions)}"
    )

    print(
        f"Changed pixels:             "
        f"{result.changed_area}"
    )

    print(
        f"Changed area ratio:         "
        f"{result.changed_area_ratio:.4%}"
    )

    print()

    for index, region in enumerate(
        result.regions,
        start=1,
    ):
        print(
            f"Region #{index}: "
            f"x={region.x}, "
            f"y={region.y}, "
            f"w={region.width}, "
            f"h={region.height}, "
            f"area={region.area}"
        )

    print()

    expected_q14_area = any(
        region.x < 200
        and region.y < 200
        and region.x + region.width > 120
        and region.y + region.height > 120
        for region in result.regions
    )

    reasonable_region_count = (
        len(result.regions) <= 4
    )

    reasonable_changed_area = (
        result.changed_area_ratio < 0.015
    )

    if (
        result.changed
        and expected_q14_area
        and reasonable_region_count
        and reasonable_changed_area
        and registration_result.inlier_ratio
        >= 0.30
    ):
        print(
            "[PASS] Inspection image registered"
        )
        print(
            "[PASS] Registration residuals suppressed"
        )
        print(
            "[PASS] Real visual change detected"
        )
        print(
            "[PASS] Changed component region localized"
        )
        print(
            "[PASS] False-positive area controlled"
        )
        print(
            "[PASS] Validation artifacts generated"
        )
        print()
        print(
            "REDGUARD v0.0.4 "
            "CHANGE DETECTION: PASS"
        )
        return 0

    print(
        "[FAIL] End-to-end change "
        "detection failed"
    )
    print()
    print(
        "REDGUARD v0.0.4 "
        "CHANGE DETECTION: FAIL"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
