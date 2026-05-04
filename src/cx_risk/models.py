"""Model pipeline factories matching the notebook baseline models."""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .config import RANDOM_STATE


def make_model_pipeline(classifier, preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]
    )


def create_model_pipelines(preprocessor, random_state: int = RANDOM_STATE):
    return {
        "Logistic Regression": make_model_pipeline(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
            ),
            preprocessor,
        ),
        "Random Forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            preprocessor,
        ),
        "HistGradientBoostingClassifier": make_model_pipeline(
            HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=random_state,
            ),
            preprocessor,
        ),
    }


def fit_with_warning_capture(model_pipeline, X_fit, y_fit, **fit_params):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model_pipeline.fit(X_fit, y_fit, **fit_params)
    return [f"{warning.category.__name__}: {warning.message}" for warning in caught_warnings]


def fit_baseline_models(pipelines, X_train, y_train):
    fitted = {}
    warnings_by_model = {}
    for name, pipeline in pipelines.items():
        fit_params = {}
        if name == "HistGradientBoostingClassifier":
            fit_params["classifier__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)
        warnings_by_model[name] = fit_with_warning_capture(pipeline, X_train, y_train, **fit_params)
        fitted[name] = pipeline
    return fitted, warnings_by_model


def predict_model_outputs(fitted_models, X):
    outputs = {}
    for name, model in fitted_models.items():
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = np.full(len(y_pred), np.nan)
        outputs[name] = {"y_pred": y_pred, "y_proba": y_proba}
    return outputs

