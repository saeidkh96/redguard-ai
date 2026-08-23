import pytest

from redguard.imaging.change_detection import ChangedRegion
from redguard.inspection.component_verifier import (
    ComponentStatus,
    ComponentVerifier,
)
from redguard.models.component import (
    BoundingBox,
    ComponentDefinition,
)


def build_components() -> list[ComponentDefinition]:
    return [
        ComponentDefinition(
            component_id="Q14",
            component_type="transistor",
            bounding_box=BoundingBox(
                x=130,
                y=130,
                width=60,
                height=35,
            ),
        ),
        ComponentDefinition(
            component_id="R27",
            component_type="resistor",
            bounding_box=BoundingBox(
                x=670,
                y=120,
                width=90,
                height=40,
            ),
        ),
        ComponentDefinition(
            component_id="C08",
            component_type="capacitor",
            bounding_box=BoundingBox(
                x=120,
                y=430,
                width=65,
                height=60,
            ),
        ),
    ]


def test_changed_region_maps_to_q14():
    components = build_components()

    regions = [
        ChangedRegion(
            x=137,
            y=136,
            width=47,
            height=24,
            area=1056,
        )
    ]

    result = ComponentVerifier(
        min_overlap_ratio=0.05
    ).verify(
        components,
        regions,
    )

    changed_ids = {
        component.component_id
        for component in result.changed_components
    }

    assert changed_ids == {"Q14"}


def test_unaffected_components_remain_unchanged():
    components = build_components()

    regions = [
        ChangedRegion(
            x=137,
            y=136,
            width=47,
            height=24,
            area=1056,
        )
    ]

    result = ComponentVerifier().verify(
        components,
        regions,
    )

    statuses = {
        component.component_id: component.status
        for component in result.components
    }

    assert statuses["Q14"] == ComponentStatus.CHANGED
    assert statuses["R27"] == ComponentStatus.UNCHANGED
    assert statuses["C08"] == ComponentStatus.UNCHANGED


def test_no_regions_means_all_components_unchanged():
    result = ComponentVerifier().verify(
        build_components(),
        [],
    )

    assert len(result.changed_components) == 0
    assert len(result.unchanged_components) == 3


def test_overlap_ratio_is_calculated():
    component = ComponentDefinition(
        component_id="Q1",
        component_type="transistor",
        bounding_box=BoundingBox(
            x=100,
            y=100,
            width=100,
            height=100,
        ),
    )

    region = ChangedRegion(
        x=100,
        y=100,
        width=50,
        height=100,
        area=5000,
    )

    result = ComponentVerifier().verify(
        [component],
        [region],
    )

    verification = result.components[0]

    assert verification.overlap_area == 5000
    assert verification.overlap_ratio == pytest.approx(0.5)


def test_region_outside_component_does_not_match():
    component = ComponentDefinition(
        component_id="Q1",
        component_type="transistor",
        bounding_box=BoundingBox(
            x=100,
            y=100,
            width=50,
            height=50,
        ),
    )

    region = ChangedRegion(
        x=300,
        y=300,
        width=40,
        height=40,
        area=1600,
    )

    result = ComponentVerifier().verify(
        [component],
        [region],
    )

    verification = result.components[0]

    assert verification.status == ComponentStatus.UNCHANGED
    assert verification.overlap_area == 0
    assert verification.overlap_ratio == 0.0


def test_duplicate_component_ids_are_rejected():
    components = [
        ComponentDefinition(
            component_id="Q14",
            component_type="transistor",
            bounding_box=BoundingBox(
                x=10,
                y=10,
                width=20,
                height=20,
            ),
        ),
        ComponentDefinition(
            component_id="Q14",
            component_type="transistor",
            bounding_box=BoundingBox(
                x=50,
                y=50,
                width=20,
                height=20,
            ),
        ),
    ]

    with pytest.raises(ValueError):
        ComponentVerifier().verify(
            components,
            [],
        )


def test_invalid_bounding_box_is_rejected():
    with pytest.raises(ValueError):
        BoundingBox(
            x=10,
            y=10,
            width=0,
            height=20,
        )


def test_invalid_overlap_threshold_is_rejected():
    with pytest.raises(ValueError):
        ComponentVerifier(
            min_overlap_ratio=1.5
        )
