"""Intervention queue evaluation utilities."""

import math

import numpy as np
import pandas as pd


def _as_1d_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def compute_lift(y_true, selected_mask) -> float:
    """Compute selected precision divided by the base positive rate."""
    y_true = _as_1d_array(y_true, "y_true").astype(int)
    selected_mask = _as_1d_array(selected_mask, "selected_mask").astype(bool)
    if len(y_true) != len(selected_mask):
        raise ValueError("y_true and selected_mask must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute lift for an empty array.")

    base_positive_rate = float(np.mean(y_true))
    if selected_mask.sum() == 0:
        return 0.0
    if base_positive_rate == 0:
        return np.nan
    return float(np.mean(y_true[selected_mask]) / base_positive_rate)


def summarize_intervention_selection(y_true, selected_mask) -> dict:
    """Summarize precision, recall, lift, and queue counts for selected rows."""
    y_true = _as_1d_array(y_true, "y_true").astype(int)
    selected_mask = _as_1d_array(selected_mask, "selected_mask").astype(bool)
    if len(y_true) != len(selected_mask):
        raise ValueError("y_true and selected_mask must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate intervention selection for an empty array.")

    n_orders = len(y_true)
    n_flagged = int(selected_mask.sum())
    total_positives = int(y_true.sum())
    true_positives = int(y_true[selected_mask].sum()) if n_flagged else 0
    false_positives = n_flagged - true_positives
    false_negatives = total_positives - true_positives
    precision = true_positives / n_flagged if n_flagged else 0.0
    recall = true_positives / total_positives if total_positives else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "n_orders": n_orders,
        "n_flagged": n_flagged,
        "pct_orders_flagged": n_flagged / n_orders,
        "base_positive_rate": float(np.mean(y_true)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift_over_random": compute_lift(y_true, selected_mask),
    }


def evaluate_top_k_intervention(y_true, y_proba, top_frac: float) -> dict:
    """Evaluate an intervention queue containing the top fraction by risk score."""
    if not 0 <= top_frac <= 1:
        raise ValueError("top_frac must be between 0 and 1.")
    y_true = _as_1d_array(y_true, "y_true").astype(int)
    y_proba = _as_1d_array(y_proba, "y_proba").astype(float)
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate intervention for an empty array.")

    n_flagged = int(math.ceil(len(y_true) * top_frac))
    selected_mask = np.zeros(len(y_true), dtype=bool)
    if n_flagged:
        ranked_indices = np.argsort(-y_proba, kind="mergesort")[:n_flagged]
        selected_mask[ranked_indices] = True

    summary = summarize_intervention_selection(y_true, selected_mask)
    summary["selection_strategy"] = "top_fraction"
    summary["top_frac"] = top_frac
    summary["probability_threshold"] = np.nan
    summary["min_selected_probability"] = float(np.min(y_proba[selected_mask])) if n_flagged else np.nan
    return summary


def build_intervention_curve(y_true, y_proba, top_fracs) -> pd.DataFrame:
    """Evaluate multiple top-fraction intervention operating points."""
    return pd.DataFrame(
        evaluate_top_k_intervention(y_true, y_proba, top_frac)
        for top_frac in top_fracs
    )


def evaluate_probability_threshold(y_true, y_proba, threshold: float) -> dict:
    """Evaluate an intervention queue selected by a probability threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    y_true = _as_1d_array(y_true, "y_true").astype(int)
    y_proba = _as_1d_array(y_proba, "y_proba").astype(float)
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate threshold for an empty array.")

    selected_mask = y_proba >= threshold
    summary = summarize_intervention_selection(y_true, selected_mask)
    summary["selection_strategy"] = "probability_threshold"
    summary["top_frac"] = np.nan
    summary["probability_threshold"] = threshold
    summary["min_selected_probability"] = float(np.min(y_proba[selected_mask])) if selected_mask.sum() else np.nan
    return summary


def evaluate_thresholds(y_true, y_proba, thresholds) -> pd.DataFrame:
    """Evaluate multiple probability-threshold operating points."""
    return pd.DataFrame(
        evaluate_probability_threshold(y_true, y_proba, threshold)
        for threshold in thresholds
    )
