from __future__ import annotations

from collections.abc import Iterable

from redguard.reference.models import ReferenceSample, ReferenceSet


class ReferenceBank:
    """In-memory store for known-normal component reference populations."""

    def __init__(self) -> None:
        self._sets: dict[str, ReferenceSet] = {}

    def add(self, sample: ReferenceSample) -> None:
        reference_set = self._sets.get(sample.component_id)

        if reference_set is None:
            reference_set = ReferenceSet(
                component_id=sample.component_id,
                component_type=sample.component_type,
            )
            self._sets[sample.component_id] = reference_set

        reference_set.add(sample)

    def extend(self, samples: Iterable[ReferenceSample]) -> None:
        for sample in samples:
            self.add(sample)

    def get(self, component_id: str) -> ReferenceSet:
        try:
            return self._sets[component_id]
        except KeyError as exc:
            raise KeyError(
                f"no reference set registered for component {component_id!r}"
            ) from exc

    def contains(self, component_id: str) -> bool:
        return component_id in self._sets

    def remove(self, component_id: str) -> None:
        if component_id not in self._sets:
            raise KeyError(
                f"no reference set registered for component {component_id!r}"
            )

        del self._sets[component_id]

    @property
    def component_count(self) -> int:
        return len(self._sets)

    @property
    def sample_count(self) -> int:
        return sum(reference_set.size for reference_set in self._sets.values())

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sets))