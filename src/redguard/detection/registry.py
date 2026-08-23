from dataclasses import dataclass

from redguard.models.component import (
    BoundingBox,
    ComponentDefinition,
)
from redguard.models.detection import ComponentDetection


@dataclass(frozen=True)
class GeneratedComponentRegistry:
    components: tuple[ComponentDefinition, ...]
    source_detector: str


def build_component_registry(
    detections: list[ComponentDetection]
    | tuple[ComponentDetection, ...],
) -> GeneratedComponentRegistry:
    """
    Convert model detections into RedGuard component definitions.

    Component IDs are deterministic within a detection result:
    transistor_001, resistor_001, etc.
    """

    if not detections:
        return GeneratedComponentRegistry(
            components=(),
            source_detector="unknown",
        )

    detector_names = {
        detection.detector_name
        for detection in detections
    }

    if len(detector_names) != 1:
        raise ValueError(
            "All detections must come from the same detector."
        )

    counters: dict[str, int] = {}

    components: list[ComponentDefinition] = []

    sorted_detections = sorted(
        detections,
        key=lambda detection: (
            detection.bounding_box.y,
            detection.bounding_box.x,
        ),
    )

    for detection in sorted_detections:
        component_type = detection.component_type

        counters[component_type] = (
            counters.get(component_type, 0)
            + 1
        )

        sequence = counters[
            component_type
        ]

        component_id = (
            f"{component_type}_{sequence:03d}"
        )

        box = detection.bounding_box

        components.append(
            ComponentDefinition(
                component_id=component_id,
                component_type=component_type,
                bounding_box=BoundingBox(
                    x=box.x,
                    y=box.y,
                    width=box.width,
                    height=box.height,
                ),
            )
        )

    return GeneratedComponentRegistry(
        components=tuple(components),
        source_detector=next(
            iter(detector_names)
        ),
    )
