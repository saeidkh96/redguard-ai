from dataclasses import dataclass
from enum import Enum

from redguard.imaging.change_detection import ChangedRegion
from redguard.models.component import (
    BoundingBox,
    ComponentDefinition,
)


class ComponentStatus(str, Enum):
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ComponentVerification:
    component_id: str
    component_type: str
    status: ComponentStatus
    overlap_area: int
    overlap_ratio: float
    confidence: float


@dataclass(frozen=True)
class ComponentVerificationResult:
    components: tuple[ComponentVerification, ...]

    @property
    def changed_components(
        self,
    ) -> tuple[ComponentVerification, ...]:
        return tuple(
            component
            for component in self.components
            if component.status == ComponentStatus.CHANGED
        )

    @property
    def unchanged_components(
        self,
    ) -> tuple[ComponentVerification, ...]:
        return tuple(
            component
            for component in self.components
            if component.status == ComponentStatus.UNCHANGED
        )


class ComponentVerifier:
    """Map localized visual changes to registered components."""

    def __init__(
        self,
        min_overlap_ratio: float = 0.05,
    ) -> None:
        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError(
                "min_overlap_ratio must be between 0 and 1."
            )

        self.min_overlap_ratio = min_overlap_ratio

    def verify(
        self,
        components: list[ComponentDefinition]
        | tuple[ComponentDefinition, ...],
        changed_regions: list[ChangedRegion]
        | tuple[ChangedRegion, ...],
    ) -> ComponentVerificationResult:
        component_ids = [
            component.component_id
            for component in components
        ]

        if len(component_ids) != len(set(component_ids)):
            raise ValueError(
                "Component IDs must be unique."
            )

        results: list[ComponentVerification] = []

        for component in components:
            overlap_area = self._calculate_total_overlap(
                component.bounding_box,
                changed_regions,
            )

            overlap_ratio = min(
                overlap_area / component.bounding_box.area,
                1.0,
            )

            changed = (
                overlap_ratio >= self.min_overlap_ratio
            )

            confidence = self._confidence(
                overlap_ratio=overlap_ratio,
                changed=changed,
            )

            results.append(
                ComponentVerification(
                    component_id=component.component_id,
                    component_type=component.component_type,
                    status=(
                        ComponentStatus.CHANGED
                        if changed
                        else ComponentStatus.UNCHANGED
                    ),
                    overlap_area=overlap_area,
                    overlap_ratio=float(overlap_ratio),
                    confidence=confidence,
                )
            )

        return ComponentVerificationResult(
            components=tuple(results)
        )

    @staticmethod
    def _calculate_total_overlap(
        component_box: BoundingBox,
        changed_regions: list[ChangedRegion]
        | tuple[ChangedRegion, ...],
    ) -> int:
        overlap_area = 0

        for region in changed_regions:
            overlap_area += ComponentVerifier._intersection_area(
                component_box,
                region,
            )

        return min(
            overlap_area,
            component_box.area,
        )

    @staticmethod
    def _intersection_area(
        component_box: BoundingBox,
        region: ChangedRegion,
    ) -> int:
        left = max(
            component_box.x,
            region.x,
        )

        top = max(
            component_box.y,
            region.y,
        )

        right = min(
            component_box.x2,
            region.x + region.width,
        )

        bottom = min(
            component_box.y2,
            region.y + region.height,
        )

        if right <= left or bottom <= top:
            return 0

        return (
            (right - left)
            * (bottom - top)
        )

    def _confidence(
        self,
        overlap_ratio: float,
        changed: bool,
    ) -> float:
        if changed:
            if self.min_overlap_ratio >= 1.0:
                return 1.0

            normalized = (
                overlap_ratio - self.min_overlap_ratio
            ) / (
                1.0 - self.min_overlap_ratio
            )

            return float(
                min(
                    1.0,
                    0.5 + 0.5 * max(0.0, normalized),
                )
            )

        if self.min_overlap_ratio == 0.0:
            return 1.0

        distance = (
            self.min_overlap_ratio - overlap_ratio
        ) / self.min_overlap_ratio

        return float(
            min(
                1.0,
                0.5 + 0.5 * max(0.0, distance),
            )
        )
