"""Scoring service used by the FastAPI app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cx_risk.explainability import add_reason_codes_to_queue
from cx_risk.scoring import (
    DEFAULT_QUEUE_MODEL,
    MODEL_REGISTRY_PATH,
    build_risk_queue,
    load_model_pipeline,
    load_model_registry,
    validate_scoring_features,
)

from .storage import ScoreStore


class RegistryUnavailableError(RuntimeError):
    pass


class UnknownModelError(ValueError):
    pass


class ScoringService:
    """Load saved model artifacts and score engineered feature payloads."""

    def __init__(
        self,
        registry_path: Path = MODEL_REGISTRY_PATH,
        store: ScoreStore | None = None,
    ):
        self.registry_path = registry_path
        self.store = store if store is not None else ScoreStore()
        self._registry = None
        self._models = {}

    @property
    def registry_available(self) -> bool:
        return self.registry_path.exists()

    def registry(self) -> dict[str, Any]:
        if not self.registry_available:
            raise RegistryUnavailableError(f"Missing model registry: {self.registry_path}")
        if self._registry is None:
            self._registry = load_model_registry(self.registry_path)
        return self._registry

    def default_model(self) -> str | None:
        if not self.registry_available:
            return None
        return self.registry().get("default_model", DEFAULT_QUEUE_MODEL)

    def model_names(self) -> list[str]:
        return [record["name"] for record in self.registry().get("models", [])]

    def model_info(self) -> dict[str, Any]:
        registry = self.registry()
        return {
            "default_model": registry.get("default_model", DEFAULT_QUEUE_MODEL),
            "feature_count": len(registry["feature_columns"]),
            "models": self.model_names(),
            "created_at": registry.get("created_at"),
        }

    def _load_model(self, model_name: str):
        if model_name not in self.model_names():
            raise UnknownModelError(f"Unknown model: {model_name}")
        if model_name not in self._models:
            self._models[model_name] = load_model_pipeline(model_name)
        return self._models[model_name]

    def score_record(
        self,
        order_id: str,
        features: dict[str, Any],
        model_name: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        registry = self.registry()
        selected_model = model_name or registry.get("default_model", DEFAULT_QUEUE_MODEL)
        model = self._load_model(selected_model)
        feature_columns = registry["feature_columns"]
        X = pd.DataFrame([features])
        validate_scoring_features(X, feature_columns)
        X = X[feature_columns]

        risk_score = float(model.predict_proba(X)[:, 1][0])
        queue = build_risk_queue(
            selected_model,
            order_ids=pd.Series([order_id]),
            X_scoring=X[feature_columns],
            risk_scores=[risk_score],
            y_true=None,
        )
        queue = add_reason_codes_to_queue(queue)
        row = queue.iloc[0]
        response = {
            "order_id": str(row["order_id"]),
            "model": str(row["model"]),
            "risk_score": float(row["risk_score"]),
            "risk_band": str(row["risk_band"]),
            "recommended_action": str(row["recommended_action"]),
            "reason_1": str(row.get("reason_1", "")),
            "reason_2": str(row.get("reason_2", "")),
            "reason_3": str(row.get("reason_3", "")),
            "reason_summary": str(row.get("reason_summary", "")),
        }
        if persist:
            self.store.insert_score(response)
        return response

    def recent_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        return self.store.recent_scores(limit=limit)
