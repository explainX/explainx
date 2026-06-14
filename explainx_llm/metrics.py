"""Model performance metrics for classification and regression."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import ModelMetrics
from .utils import predict, predict_scores


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y_true: Any,
    problem_type: str,
) -> ModelMetrics:
    y_true = np.asarray(y_true)
    y_pred = predict(model, X)

    if problem_type == "classification":
        return _classification_metrics(model, X, y_true, y_pred)
    return _regression_metrics(y_true, y_pred)


def _classification_metrics(model, X, y_true, y_pred) -> ModelMetrics:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )

    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }

    # ROC-AUC needs scores and only makes sense for the binary case here.
    if avg == "binary":
        scores = predict_scores(model, X, "classification")
        if scores is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            except Exception:
                pass

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return ModelMetrics(
        problem_type="classification",
        metrics=metrics,
        confusion_matrix=cm,
        labels=labels,
    )


def _regression_metrics(y_true, y_pred) -> ModelMetrics:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
    }
    # MAPE only when no zero targets, to avoid divide-by-zero blowups.
    if not np.any(y_true == 0):
        metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return ModelMetrics(problem_type="regression", metrics=metrics)
