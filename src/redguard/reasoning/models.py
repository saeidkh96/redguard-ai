from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ReasoningContext:
    component_id: str
    component_type: str
    decision: str
    severity: str
    risk_score: float
    confidence: float
    evidence: tuple[dict[str, Any], ...]


@dataclass(slots=True, frozen=True)
class ReasoningOutput:
    explanation: str
    observations: tuple[str, ...]
    provider: str