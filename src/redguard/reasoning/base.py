from __future__ import annotations

from abc import ABC, abstractmethod

from redguard.reasoning.models import (
    ReasoningContext,
    ReasoningOutput,
)


class ReasoningProvider(ABC):
    @abstractmethod
    def explain(
        self,
        context: ReasoningContext,
    ) -> ReasoningOutput:
        raise NotImplementedError