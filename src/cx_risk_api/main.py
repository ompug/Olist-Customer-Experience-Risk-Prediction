"""FastAPI app for the Olist CX risk scoring service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .schemas import (
    BatchScoreRequest,
    HealthResponse,
    ModelInfoResponse,
    ScoreHistoryResponse,
    ScoreRequest,
    ScoreResponse,
)
from .service import RegistryUnavailableError, ScoringService, UnknownModelError


def create_app(scoring_service: ScoringService | None = None) -> FastAPI:
    service = scoring_service if scoring_service is not None else ScoringService()
    api = FastAPI(
        title="Olist CX Risk Scoring API",
        version="0.1.0",
        description="Local scoring API for the Olist customer-experience risk project.",
    )

    @api.get("/health", response_model=HealthResponse)
    def health():
        default_model = service.default_model() if service.registry_available else None
        return {
            "status": "ok" if service.registry_available else "missing_registry",
            "registry_available": service.registry_available,
            "default_model": default_model,
        }

    @api.get("/models", response_model=ModelInfoResponse)
    def models():
        try:
            return service.model_info()
        except RegistryUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest):
        try:
            return service.score_record(
                order_id=request.order_id,
                features=request.features,
                model_name=request.model_name,
            )
        except RegistryUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except UnknownModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/score-batch", response_model=list[ScoreResponse])
    def score_batch(request: BatchScoreRequest):
        responses = []
        for record in request.records:
            selected_model = record.model_name or request.model_name
            try:
                responses.append(
                    service.score_record(
                        order_id=record.order_id,
                        features=record.features,
                        model_name=selected_model,
                    )
                )
            except RegistryUnavailableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except UnknownModelError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{record.order_id}: {exc}") from exc
        return responses

    @api.get("/scores", response_model=ScoreHistoryResponse)
    def scores(limit: int = Query(default=50, ge=1, le=500)):
        return {"records": service.recent_scores(limit=limit)}

    return api


app = create_app()

