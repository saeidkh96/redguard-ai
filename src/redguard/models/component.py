from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(
                "Bounding box coordinates cannot be negative."
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Bounding box dimensions must be positive."
            )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class ComponentDefinition:
    component_id: str
    component_type: str
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id cannot be empty."
            )

        if not self.component_type.strip():
            raise ValueError(
                "component_type cannot be empty."
            )
