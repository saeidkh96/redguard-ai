from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class InspectionRequest(BaseModel):
    component_id: str = Field(min_length=1)
    component_type: str = Field(min_length=1)
    change_score: float = Field(ge=0, le=1)
    fingerprint_similarity: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    edge_difference: float = Field(ge=0, le=1)
    texture_difference: float = Field(ge=0, le=1)
    reference_consensus: float = Field(ge=0, le=1)


class InspectionResponse(BaseModel):
    inspection_id: str
    component_id: str
    component_type: str
    decision: str
    severity: str
    risk_score: float
    confidence: float
    explanation: str
    evidence: list[dict[str, Any]]
    artifacts: list[str]
    created_at: str
