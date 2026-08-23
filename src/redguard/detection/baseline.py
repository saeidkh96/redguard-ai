import cv2
import numpy as np

from redguard.detection.base import ComponentDetector
from redguard.imaging.validation import validate_image
from redguard.models.component import BoundingBox
from redguard.models.detection import ComponentDetection


class BaselineComponentDetector(ComponentDetector):
    """
    Deterministic contour-based baseline detector.

    This detector is used to validate the automatic detection
    architecture before introducing a trained object detector.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 20000,
    ) -> None:
        if min_area <= 0:
            raise ValueError(
                "min_area must be positive."
            )

        if max_area <= min_area:
            raise ValueError(
                "max_area must be greater than min_area."
            )

        self.min_area = min_area
        self.max_area = max_area

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[ComponentDetection, ...]:
        validate_image(image)

        gray = self._to_grayscale(image)

        _, binary = cv2.threshold(
            gray,
            150,
            255,
            cv2.THRESH_BINARY,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: list[ComponentDetection] = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            component_type = self._classify_shape(
                width,
                height,
            )

            confidence = self._baseline_confidence(
                area=area,
                width=width,
                height=height,
            )

            detections.append(
                ComponentDetection(
                    component_type=component_type,
                    bounding_box=BoundingBox(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ),
                    confidence=confidence,
                    detector_name="baseline-contour",
                )
            )

        detections.sort(
            key=lambda item: (
                item.bounding_box.y,
                item.bounding_box.x,
            )
        )

        return tuple(detections)

    @staticmethod
    def _classify_shape(
        width: int,
        height: int,
    ) -> str:
        aspect_ratio = width / height

        if aspect_ratio >= 2.0:
            return "resistor"

        if aspect_ratio <= 0.75:
            return "capacitor"

        if width >= 70 and height >= 50:
            return "integrated_circuit"

        return "transistor"

    @staticmethod
    def _baseline_confidence(
        area: float,
        width: int,
        height: int,
    ) -> float:
        rectangular_area = width * height

        if rectangular_area <= 0:
            return 0.0

        fill_ratio = area / rectangular_area

        return float(
            max(
                0.0,
                min(
                    1.0,
                    0.5 + 0.5 * fill_ratio,
                ),
            )
        )

    @staticmethod
    def _to_grayscale(
        image: np.ndarray,
    ) -> np.ndarray:
        if image.ndim == 2:
            return image

        if image.shape[2] == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )
