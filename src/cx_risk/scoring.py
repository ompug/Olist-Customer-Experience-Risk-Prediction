"""Production-style model artifact and risk-queue helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .preprocessing import assert_no_forbidden_columns
from .utils import ensure_directory

DEFAULT_QUEUE_MODEL = "HistGradientBoostingClassifier"
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"

QUEUE_CONTEXT_COLUMNS = [
    "order_purchase_timestamp",
    "customer_state",
    "primary_category",
    "estimated_delivery_days",
    "freight_share",
    "multi_seller_flag",
    "high_installment_flag",
    "expensive_order_flag",
    "seller_prior_order_count",
    "seller_prior_low_review_rate",
    "category_prior_low_review_rate",
]

RISK_QUEUE_COLUMNS = [
    "model",
    "queue_rank",
    "order_id",
    "risk_score",
    "risk_band",
    "recommended_action",
    *QUEUE_CONTEXT_COLUMNS,
    "actual_low_review",
    "is_true_positive",
]


def model_slug(model_name: str) -> str:
    """Convert a display model name into a stable artifact filename stem."""
    snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", model_name)
    slug = re.sub(r"[^a-z0-9]+", "_", snake_name.lower()).strip("_")
    if not slug:
        raise ValueError("Model name must contain at least one alphanumeric character.")
    return slug


def model_artifact_path(model_name: str, model_dir: Path = MODEL_DIR) -> Path:
    """Return the local artifact path for a persisted model pipeline."""
    return model_dir / f"{model_slug(model_name)}.joblib"


def save_model_pipeline(model_name: str, pipeline, model_dir: Path = MODEL_DIR) -> Path:
    """Persist a fitted sklearn pipeline and return its artifact path."""
    ensure_directory(model_dir)
    path = model_artifact_path(model_name, model_dir=model_dir)
    joblib.dump(pipeline, path)
    return path


def load_model_pipeline(model_name: str, model_dir: Path = MODEL_DIR):
    """Load a fitted sklearn pipeline by display model name."""
    path = model_artifact_path(model_name, model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact for {model_name}: {path}")
    return joblib.load(path)


def save_model_registry(registry: dict[str, Any], path: Path = MODEL_REGISTRY_PATH) -> Path:
    """Save model registry metadata as JSON."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, default=str)
    return path


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load model registry metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model registry: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for artifact metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_scoring_features(X: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate that scoring inputs include only the expected leakage-safe feature columns."""
    missing = [column for column in feature_columns if column not in X.columns]
    if missing:
        raise ValueError(f"Scoring input is missing feature columns: {missing}")
    unexpected = sorted(set(X.columns) - set(feature_columns))
    if unexpected:
        raise ValueError(f"Scoring input contains unexpected columns: {unexpected}")
    assert_no_forbidden_columns(feature_columns, include_raw_ids=True)


def assign_risk_band(risk_scores) -> pd.Series:
    """Assign product-facing risk bands from class-1 probabilities."""
    scores = pd.Series(risk_scores, dtype=float)
    bands = np.select(
        [
            scores >= 0.75,
            scores >= 0.50,
            scores >= 0.25,
        ],
        ["critical", "high", "medium"],
        default="low",
    )
    return pd.Series(bands, index=scores.index)


def recommend_intervention(row: pd.Series) -> str:
    """Return a simple deterministic action label for a queued order."""
    if row["risk_band"] not in {"critical", "high"}:
        if row["risk_band"] == "medium":
            return "Monitor"
        return "No action"

    estimated_days = row.get("estimated_delivery_days")
    seller_rate = row.get("seller_prior_low_review_rate")
    seller_count = row.get("seller_prior_order_count")

    if pd.notna(estimated_days) and estimated_days >= 20:
        return "Shipping check"
    if (
        pd.notna(seller_rate)
        and pd.notna(seller_count)
        and seller_rate >= 0.20
        and seller_count >= 5
    ):
        return "Seller escalation"
    return "Proactive customer outreach"


def build_risk_queue(
    model_name: str,
    order_ids: pd.Series,
    X_scoring: pd.DataFrame,
    risk_scores,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a ranked risk queue for one model from scoring features and probabilities."""
    if len(order_ids) != len(X_scoring):
        raise ValueError("order_ids and X_scoring must have the same length.")

    queue = pd.DataFrame(
        {
            "model": model_name,
            "order_id": order_ids.reset_index(drop=True),
            "risk_score": pd.Series(risk_scores, dtype=float).reset_index(drop=True),
        }
    )
    for column in QUEUE_CONTEXT_COLUMNS:
        queue[column] = (
            X_scoring[column].reset_index(drop=True)
            if column in X_scoring.columns
            else np.nan
        )

    queue["risk_band"] = assign_risk_band(queue["risk_score"])
    queue["recommended_action"] = queue.apply(recommend_intervention, axis=1)

    if y_true is not None:
        if len(y_true) != len(queue):
            raise ValueError("y_true and queue must have the same length.")
        queue["actual_low_review"] = y_true.reset_index(drop=True).astype(int)
        queue["is_true_positive"] = (
            queue["actual_low_review"].eq(1)
            & queue["risk_band"].isin(["critical", "high"])
        ).astype(int)
    else:
        queue["actual_low_review"] = np.nan
        queue["is_true_positive"] = np.nan

    queue = queue.sort_values("risk_score", ascending=False, kind="mergesort").reset_index(drop=True)
    queue.insert(1, "queue_rank", np.arange(1, len(queue) + 1))
    return queue[RISK_QUEUE_COLUMNS]


def build_combined_risk_queue(
    model_scores: dict[str, np.ndarray],
    order_ids: pd.Series,
    X_scoring: pd.DataFrame,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Build one risk queue table containing every scored model."""
    frames = [
        build_risk_queue(model_name, order_ids, X_scoring, risk_scores, y_true=y_true)
        for model_name, risk_scores in model_scores.items()
    ]
    return pd.concat(frames, ignore_index=True)
