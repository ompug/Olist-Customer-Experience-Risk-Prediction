"""API request and response schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    order_id: str = Field(..., min_length=1)
    features: dict[str, Any]
    model_name: Optional[str] = None


class BatchScoreRequest(BaseModel):
    records: list[ScoreRequest]
    model_name: Optional[str] = None


class ScoreResponse(BaseModel):
    order_id: str
    model: str
    risk_score: float
    risk_band: str
    recommended_action: str
    reason_1: str
    reason_2: str
    reason_3: str
    reason_summary: str


class HealthResponse(BaseModel):
    status: str
    registry_available: bool
    default_model: Optional[str] = None


class ModelInfoResponse(BaseModel):
    default_model: str
    feature_count: int
    models: list[str]
    created_at: Optional[str] = None


class ScoreHistoryResponse(BaseModel):
    records: list[dict[str, Any]]

