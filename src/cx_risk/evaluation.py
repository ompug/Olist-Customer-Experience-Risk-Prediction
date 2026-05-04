"""Classification evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(model_name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_class_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc_average_precision": average_precision_score(y_true, y_proba),
        "true_negatives": cm[0, 0],
        "false_positives": cm[0, 1],
        "false_negatives": cm[1, 0],
        "true_positives": cm[1, 1],
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def create_model_comparison_table(y_true, model_outputs):
    records = [
        compute_classification_metrics(name, y_true, outputs["y_pred"], outputs["y_proba"])
        for name, outputs in model_outputs.items()
    ]
    return pd.DataFrame(records).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def classification_reports(y_true, model_outputs):
    reports = {}
    for name, outputs in model_outputs.items():
        report = classification_report(
            y_true,
            outputs["y_pred"],
            labels=[0, 1],
            target_names=["not_low_review", "low_review"],
            output_dict=True,
            zero_division=0,
        )
        reports[name] = pd.DataFrame(report).T
    return reports


def confusion_matrices(y_true, model_outputs):
    return {
        name: {
            "raw": confusion_matrix(y_true, outputs["y_pred"], labels=[0, 1]),
            "normalized": confusion_matrix(y_true, outputs["y_pred"], labels=[0, 1], normalize="true"),
        }
        for name, outputs in model_outputs.items()
    }


def roc_curve_data(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds})


def precision_recall_curve_data(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": threshold_values})

