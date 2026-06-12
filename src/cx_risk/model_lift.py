"""Model and feature lift experiment helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


TOP_K_FRACTIONS = [0.01, 0.05, 0.10, 0.20]


def _validate_binary_scores(y_true, y_proba) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    if y_true_array.ndim != 1 or y_proba_array.ndim != 1:
        raise ValueError("y_true and y_proba must be one-dimensional.")
    if len(y_true_array) != len(y_proba_array):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_true_array) == 0:
        raise ValueError("Cannot compute ranking metrics for empty arrays.")
    return y_true_array, y_proba_array


def ranking_metrics_at_fraction(y_true, y_proba, top_frac: float) -> dict:
    """Compute precision, recall, and lift for the top fraction by predicted risk."""
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must be greater than 0 and at most 1.")
    y_true_array, y_proba_array = _validate_binary_scores(y_true, y_proba)
    n_flagged = int(math.ceil(len(y_true_array) * top_frac))
    selected = np.zeros(len(y_true_array), dtype=bool)
    selected[np.argsort(-y_proba_array, kind="mergesort")[:n_flagged]] = True

    total_positives = int(y_true_array.sum())
    true_positives = int(y_true_array[selected].sum())
    precision = true_positives / n_flagged if n_flagged else 0.0
    recall = true_positives / total_positives if total_positives else 0.0
    base_rate = float(y_true_array.mean())
    lift = precision / base_rate if base_rate else np.nan
    return {
        "top_frac": top_frac,
        "n_flagged": n_flagged,
        "true_positives": true_positives,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "base_positive_rate": base_rate,
    }


def build_topk_ranking_table(y_true, y_proba, top_fracs=None) -> pd.DataFrame:
    """Build top-k ranking metrics over multiple queue sizes."""
    if top_fracs is None:
        top_fracs = TOP_K_FRACTIONS
    return pd.DataFrame(
        ranking_metrics_at_fraction(y_true, y_proba, top_frac)
        for top_frac in top_fracs
    )
