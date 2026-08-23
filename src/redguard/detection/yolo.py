from pathlib import Path

import numpy as np
from ultralytics import YOLO

from redguard.detection.base import ComponentDetector
from redguard.imaging.validation import validate_image
from redguard.models.component import BoundingBox
from redguard.models.detection import ComponentDetection


class YoloComponentDetector(ComponentDetector):
    """YOLO-backed component detector for RedGuard AI."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model does not exist: "
                f"{self.model_path}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[ComponentDetection, ...]:
        validate_image(image)

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[ComponentDetection] = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                xyxy = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                confidence = float(
                    box.conf[0]
                    .detach()
                    .cpu()
                    .item()
                )

                class_id = int(
                    box.cls[0]
                    .detach()
                    .cpu()
                    .item()
                )

                x1, y1, x2, y2 = [
                    int(round(value))
                    for value in xyxy
                ]

                width = max(
                    1,
                    x2 - x1,
                )

                height = max(
                    1,
                    y2 - y1,
                )

                class_name = str(
                    result.names[class_id]
                )

                detections.append(
                    ComponentDetection(
                        component_type=class_name,
                        bounding_box=BoundingBox(
                            x=max(0, x1),
                            y=max(0, y1),
                            width=width,
                            height=height,
                        ),
                        confidence=confidence,
                        detector_name="yolo11n-redguard",
                    )
                )

        detections.sort(
            key=lambda item: (
                item.bounding_box.y,
                item.bounding_box.x,
            )
        )

        return tuple(detections)
