import pytest

from redguard.detection.registry import (
    build_component_registry,
)
from redguard.models.component import BoundingBox
from redguard.models.detection import (
    ComponentDetection,
)


def build_detection(
    component_type: str,
    x: int,
    y: int,
) -> ComponentDetection:
    return ComponentDetection(
        component_type=component_type,
        bounding_box=BoundingBox(
            x=x,
            y=y,
            width=50,
            height=30,
        ),
        confidence=0.9,
        detector_name="test-detector",
    )


def test_builds_component_registry():
    detections = [
        build_detection(
            "transistor",
            100,
            100,
        ),
        build_detection(
            "resistor",
            200,
            100,
        ),
    ]

    registry = build_component_registry(
        detections
    )

    assert len(registry.components) == 2
    assert (
        registry.source_detector
        == "test-detector"
    )


def test_generates_deterministic_ids():
    detections = [
        build_detection(
            "transistor",
            100,
            100,
        ),
        build_detection(
            "transistor",
            200,
            100,
        ),
    ]

    registry = build_component_registry(
        detections
    )

    ids = [
        component.component_id
        for component
        in registry.components
    ]

    assert ids == [
        "transistor_001",
        "transistor_002",
    ]


def test_registry_order_is_spatially_stable():
    detections = [
        build_detection(
            "resistor",
            300,
            200,
        ),
        build_detection(
            "resistor",
            100,
            100,
        ),
    ]

    registry = build_component_registry(
        detections
    )

    assert (
        registry.components[0]
        .bounding_box.y
        <= registry.components[1]
        .bounding_box.y
    )


def test_empty_detection_list():
    registry = build_component_registry(
        []
    )

    assert registry.components == ()
    assert (
        registry.source_detector
        == "unknown"
    )


def test_mixed_detector_sources_rejected():
    first = ComponentDetection(
        component_type="transistor",
        bounding_box=BoundingBox(
            x=10,
            y=10,
            width=20,
            height=20,
        ),
        confidence=0.9,
        detector_name="detector-a",
    )

    second = ComponentDetection(
        component_type="resistor",
        bounding_box=BoundingBox(
            x=50,
            y=50,
            width=20,
            height=20,
        ),
        confidence=0.8,
        detector_name="detector-b",
    )

    with pytest.raises(ValueError):
        build_component_registry(
            [first, second]
        )
