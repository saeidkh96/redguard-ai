from dataclasses import dataclass

import cv2
import numpy as np

from redguard.imaging.validation import validate_image


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the RedGuard preprocessing pipeline."""

    target_width: int | None = None
    target_height: int | None = None
    grayscale: bool = True
    denoise: bool = True
    normalize_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


class ImagePreprocessor:
    """Standardize images before registration and visual comparison."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        validate_image(image)

        result = image.copy()

        if self.config.target_width or self.config.target_height:
            result = self._resize_preserving_aspect_ratio(result)

        if self.config.grayscale:
            result = self._to_grayscale(result)

        if self.config.denoise:
            result = self._denoise(result)

        if self.config.normalize_contrast:
            result = self._normalize_contrast(result)

        return result

    def _resize_preserving_aspect_ratio(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        target_width = self.config.target_width
        target_height = self.config.target_height

        if target_width and target_height:
            scale = min(
                target_width / width,
                target_height / height,
            )
        elif target_width:
            scale = target_width / width
        elif target_height:
            scale = target_height / height
        else:
            return image

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image

        channels = image.shape[2]

        if channels == 1:
            return image[:, :, 0]

        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        return image

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            image,
            (3, 3),
            sigmaX=0,
        )

    def _normalize_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if image.ndim == 2:
            return self._apply_clahe(image)

        # Preserve color information when grayscale processing is disabled.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)

        normalized_lightness = self._apply_clahe(lightness)

        normalized_lab = cv2.merge(
            (
                normalized_lightness,
                channel_a,
                channel_b,
            )
        )

        return cv2.cvtColor(
            normalized_lab,
            cv2.COLOR_LAB2BGR,
        )

    def _apply_clahe(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(
                self.config.clahe_grid_size,
                self.config.clahe_grid_size,
            ),
        )

        return clahe.apply(image)
