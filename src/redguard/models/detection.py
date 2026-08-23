from dataclasses import dataclass

from redguard.models.component import BoundingBox


@dataclass(frozen=True)
class ComponentDetection:
    component_type: str
    bounding_box: BoundingBox
    confidence: float
    detector_name: str

    def __post_init__(self) -> None:
        if not self.component_type.strip():
            raise ValueError(
                "component_type cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if not self.detector_name.strip():
            raise ValueError(
                "detector_name cannot be empty."
            )
