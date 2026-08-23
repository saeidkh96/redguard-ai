from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from redguard.imaging.validation import validate_image


@dataclass(frozen=True)
class ChangedRegion:
    x: int
    y: int
    width: int
    height: int
    area: int


@dataclass(frozen=True)
class ChangeDetectionResult:
    changed: bool
    similarity: float
    changed_area: int
    changed_area_ratio: float
    regions: tuple[ChangedRegion, ...]
    difference_map: np.ndarray
    binary_mask: np.ndarray


class ChangeDetector:
    """Detect meaningful visual changes while suppressing registration residuals."""

    def __init__(
        self,
        threshold: int = 35,
        ssim_threshold: int = 25,
        min_region_area: int = 40,
        morphology_kernel_size: int = 3,
        comparison_blur_size: int = 5,
        border_margin: int = 8,
    ) -> None:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be between 0 and 255.")

        if not 0 <= ssim_threshold <= 255:
            raise ValueError(
                "ssim_threshold must be between 0 and 255."
            )

        if min_region_area <= 0:
            raise ValueError(
                "min_region_area must be positive."
            )

        if morphology_kernel_size <= 0:
            raise ValueError(
                "morphology_kernel_size must be positive."
            )

        if comparison_blur_size <= 0:
            raise ValueError(
                "comparison_blur_size must be positive."
            )

        if comparison_blur_size % 2 == 0:
            raise ValueError(
                "comparison_blur_size must be odd."
            )

        if border_margin < 0:
            raise ValueError(
                "border_margin cannot be negative."
            )

        self.threshold = threshold
        self.ssim_threshold = ssim_threshold
        self.min_region_area = min_region_area
        self.morphology_kernel_size = morphology_kernel_size
        self.comparison_blur_size = comparison_blur_size
        self.border_margin = border_margin

    def detect(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> ChangeDetectionResult:
        validate_image(reference)
        validate_image(inspection)

        if reference.shape[:2] != inspection.shape[:2]:
            raise ValueError(
                "Reference and inspection images must have "
                "identical spatial dimensions."
            )

        reference_gray = self._to_grayscale(reference)
        inspection_gray = self._to_grayscale(inspection)

        effective_mask = self._prepare_valid_mask(
            reference_gray.shape,
            valid_mask,
        )

        reference_filtered = cv2.GaussianBlur(
            reference_gray,
            (
                self.comparison_blur_size,
                self.comparison_blur_size,
            ),
            sigmaX=0,
        )

        inspection_filtered = cv2.GaussianBlur(
            inspection_gray,
            (
                self.comparison_blur_size,
                self.comparison_blur_size,
            ),
            sigmaX=0,
        )

        similarity, ssim_map = structural_similarity(
            reference_filtered,
            inspection_filtered,
            data_range=255,
            full=True,
        )

        ssim_difference = np.clip(
            (1.0 - ssim_map) * 255.0,
            0,
            255,
        ).astype(np.uint8)

        absolute_difference = cv2.absdiff(
            reference_filtered,
            inspection_filtered,
        )

        _, absolute_mask = cv2.threshold(
            absolute_difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )

        _, ssim_mask = cv2.threshold(
            ssim_difference,
            self.ssim_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # A region must be supported by BOTH pixel difference
        # and structural difference.
        binary_mask = cv2.bitwise_and(
            absolute_mask,
            ssim_mask,
        )

        difference_map = cv2.min(
            absolute_difference,
            ssim_difference,
        )

        binary_mask = cv2.bitwise_and(
            binary_mask,
            effective_mask,
        )

        difference_map = cv2.bitwise_and(
            difference_map,
            difference_map,
            mask=effective_mask,
        )

        kernel = np.ones(
            (
                self.morphology_kernel_size,
                self.morphology_kernel_size,
            ),
            dtype=np.uint8,
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        binary_mask = cv2.bitwise_and(
            binary_mask,
            effective_mask,
        )

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions: list[ChangedRegion] = []
        filtered_mask = np.zeros_like(binary_mask)

        for contour in contours:
            area = int(
                round(
                    cv2.contourArea(contour)
                )
            )

            if area < self.min_region_area:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            regions.append(
                ChangedRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    area=area,
                )
            )

            cv2.drawContours(
                filtered_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )

        regions.sort(
            key=lambda region: region.area,
            reverse=True,
        )

        changed_area = int(
            cv2.countNonZero(filtered_mask)
        )

        valid_area = int(
            cv2.countNonZero(effective_mask)
        )

        changed_area_ratio = (
            changed_area / valid_area
            if valid_area > 0
            else 0.0
        )

        return ChangeDetectionResult(
            changed=bool(regions),
            similarity=float(similarity),
            changed_area=changed_area,
            changed_area_ratio=float(
                changed_area_ratio
            ),
            regions=tuple(regions),
            difference_map=difference_map,
            binary_mask=filtered_mask,
        )

    def _prepare_valid_mask(
        self,
        shape: tuple[int, int],
        valid_mask: np.ndarray | None,
    ) -> np.ndarray:
        height, width = shape

        if valid_mask is None:
            mask = np.full(
                (height, width),
                255,
                dtype=np.uint8,
            )
        else:
            if valid_mask.shape != shape:
                raise ValueError(
                    "valid_mask must match image "
                    "spatial dimensions."
                )

            mask = valid_mask.astype(
                np.uint8,
                copy=True,
            )

        if self.border_margin > 0:
            margin = self.border_margin

            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0

        return mask

    @staticmethod
    def _to_grayscale(
        image: np.ndarray,
    ) -> np.ndarray:
        if image.ndim == 2:
            return image

        channels = image.shape[2]

        if channels == 1:
            return image[:, :, 0]

        if channels == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )
