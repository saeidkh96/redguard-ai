from pathlib import Path

import cv2
import numpy as np

from redguard.imaging.registration import ImageRegistrationEngine


ARTIFACT_DIR = Path("artifacts/registration")


def build_reference() -> np.ndarray:
    image = np.full((600, 900, 3), 25, dtype=np.uint8)

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
            (top_left[0], top_left[1] - 10),
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


def transform(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    rotation = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        4.0,
        1.0,
    )

    rotation[0, 2] += 18
    rotation[1, 2] -= 12

    return cv2.warpAffine(
        image,
        rotation,
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


def difference_image(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    return cv2.absdiff(first, second)


def main() -> int:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = build_reference()
    inspection = transform(reference)

    before_error = mean_absolute_difference(
        reference,
        inspection,
    )

    engine = ImageRegistrationEngine()

    result = engine.register(
        reference,
        inspection,
    )

    after_error = mean_absolute_difference(
        reference,
        result.aligned_image,
    )

    diff_before = difference_image(
        reference,
        inspection,
    )

    diff_after = difference_image(
        reference,
        result.aligned_image,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "reference.png"),
        reference,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "inspection_transformed.png"),
        inspection,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "inspection_aligned.png"),
        result.aligned_image,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "difference_before.png"),
        diff_before,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "difference_after.png"),
        diff_after,
    )

    improvement = (
        (before_error - after_error)
        / before_error
        * 100.0
    )

    print("RedGuard AI Registration Validation")
    print("=" * 37)
    print(f"Matches total:    {result.matches_total}")
    print(f"Matches used:     {result.matches_used}")
    print(f"Inliers:          {result.inliers}")
    print(f"Inlier ratio:     {result.inlier_ratio:.3f}")
    print()
    print(f"Error before:     {before_error:.3f}")
    print(f"Error after:      {after_error:.3f}")
    print(f"Improvement:      {improvement:.2f}%")
    print()

    if (
        after_error < before_error
        and result.inliers >= 4
        and result.inlier_ratio >= 0.30
    ):
        print("[PASS] Registration reduced alignment error")
        print("[PASS] Geometric inliers validated")
        print("[PASS] Registration artifacts generated")
        print()
        print("REDGUARD v0.0.3 REGISTRATION: PASS")
        return 0

    print("[FAIL] Registration validation failed")
    print()
    print("REDGUARD v0.0.3 REGISTRATION: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
