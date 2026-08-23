from pathlib import Path

import cv2
import numpy as np

from redguard.imaging.preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)


ARTIFACT_DIR = Path("artifacts/preprocessing")


def build_synthetic_board() -> np.ndarray:
    image = np.full((600, 900, 3), 35, dtype=np.uint8)

    # PCB body
    cv2.rectangle(image, (80, 70), (820, 530), (60, 115, 60), -1)

    # IC
    cv2.rectangle(image, (330, 210), (570, 390), (25, 25, 25), -1)
    cv2.putText(
        image,
        "RG-IC01",
        (370, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    # Components
    for x in (150, 240, 660, 750):
        cv2.rectangle(image, (x, 150), (x + 45, 180), (180, 180, 180), -1)
        cv2.rectangle(image, (x, 420), (x + 45, 450), (180, 180, 180), -1)

    # Mounting holes
    for point in ((120, 110), (780, 110), (120, 490), (780, 490)):
        cv2.circle(image, point, 18, (15, 15, 15), -1)

    return image


def modify_illumination(image: np.ndarray) -> np.ndarray:
    darker = cv2.convertScaleAbs(
        image,
        alpha=0.72,
        beta=20,
    )

    gradient = np.tile(
        np.linspace(0.75, 1.20, image.shape[1], dtype=np.float32),
        (image.shape[0], 1),
    )

    result = darker.astype(np.float32)
    result *= gradient[:, :, None]

    return np.clip(result, 0, 255).astype(np.uint8)


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


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    reference = build_synthetic_board()
    altered_light = modify_illumination(reference)

    config = PreprocessingConfig(
        grayscale=True,
        denoise=True,
        normalize_contrast=True,
    )

    preprocessor = ImagePreprocessor(config)

    reference_processed = preprocessor.preprocess(reference)
    altered_processed = preprocessor.preprocess(altered_light)

    raw_reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY,
    )
    raw_altered_gray = cv2.cvtColor(
        altered_light,
        cv2.COLOR_BGR2GRAY,
    )

    raw_difference = mean_absolute_difference(
        raw_reference_gray,
        raw_altered_gray,
    )

    processed_difference = mean_absolute_difference(
        reference_processed,
        altered_processed,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "reference.png"),
        reference,
    )
    cv2.imwrite(
        str(ARTIFACT_DIR / "illumination_changed.png"),
        altered_light,
    )
    cv2.imwrite(
        str(ARTIFACT_DIR / "reference_preprocessed.png"),
        reference_processed,
    )
    cv2.imwrite(
        str(ARTIFACT_DIR / "illumination_preprocessed.png"),
        altered_processed,
    )

    print("RedGuard AI Preprocessing Validation")
    print("=" * 38)
    print(f"Raw illumination difference:       {raw_difference:.3f}")
    print(f"Processed illumination difference: {processed_difference:.3f}")
    print()

    if processed_difference < raw_difference:
        print("[PASS] Illumination sensitivity reduced")
        print("[PASS] Preprocessing artifacts generated")
        print()
        print("REDGUARD v0.0.2 PREPROCESSING: PASS")
        return 0

    print("[FAIL] Preprocessing did not reduce illumination sensitivity")
    print()
    print("REDGUARD v0.0.2 PREPROCESSING: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
