"""Conformal prediction: distribution-free uncertainty quantification.

Split (inductive) conformal prediction wraps *any* fitted model and turns its
point predictions into sets/intervals with a finite-sample coverage guarantee:
the true label lands in the prediction set with probability at least 1 - alpha,
with no distributional assumptions. It is the state-of-the-art, model-agnostic
way to attach honest uncertainty to predictions -- exactly what an LLM agent
needs to know how much to trust a forecast before acting on it.

Workflow: hold out a calibration split, compute nonconformity scores, take the
conformal quantile ``qhat``, then form sets/intervals for new points.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import ConformalResult
from .utils import predict


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    # Finite-sample correction: the ceil((n+1)(1-alpha))/n quantile.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_classification(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: Any,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
    y_test: Optional[Any] = None,
) -> ConformalResult:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Conformal classification needs a model with predict_proba.")

    classes = list(getattr(model, "classes_", []))
    proba_cal = np.asarray(model.predict_proba(X_calib))
    y_calib = np.asarray(y_calib)
    class_index = {c: i for i, c in enumerate(classes)}

    # Nonconformity = 1 - predicted probability of the true class.
    true_scores = np.array(
        [1 - proba_cal[i, class_index[y]] for i, y in enumerate(y_calib)]
    )
    qhat = _conformal_quantile(true_scores, alpha)

    proba_test = np.asarray(model.predict_proba(X_test))
    sets: list[list[Any]] = []
    for row in proba_test:
        keep = [classes[i] for i in range(len(classes)) if (1 - row[i]) <= qhat]
        if not keep:  # never return an empty set; fall back to the argmax
            keep = [classes[int(np.argmax(row))]]
        sets.append([_native(c) for c in keep])

    avg_size = float(np.mean([len(s) for s in sets]))
    coverage = None
    if y_test is not None:
        y_test = np.asarray(y_test)
        coverage = float(np.mean([y_test[i] in sets[i] for i in range(len(sets))]))

    return ConformalResult(
        problem_type="classification",
        alpha=alpha,
        coverage_target=1 - alpha,
        qhat=qhat,
        prediction_sets=sets,
        average_set_size=avg_size,
        empirical_coverage=coverage,
    )


def conformal_regression(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: Any,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
    y_test: Optional[Any] = None,
) -> ConformalResult:
    y_calib = np.asarray(y_calib, dtype=float)
    residuals = np.abs(y_calib - predict(model, X_calib).astype(float))
    qhat = _conformal_quantile(residuals, alpha)

    preds = predict(model, X_test).astype(float)
    intervals = [[float(p - qhat), float(p + qhat)] for p in preds]

    coverage = None
    if y_test is not None:
        y_test = np.asarray(y_test, dtype=float)
        coverage = float(
            np.mean([intervals[i][0] <= y_test[i] <= intervals[i][1] for i in range(len(preds))])
        )

    return ConformalResult(
        problem_type="regression",
        alpha=alpha,
        coverage_target=1 - alpha,
        qhat=qhat,
        intervals=intervals,
        average_interval_width=float(2 * qhat),
        empirical_coverage=coverage,
    )


def conformal_predict(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: Any,
    X_test: pd.DataFrame,
    problem_type: str,
    alpha: float = 0.1,
    y_test: Optional[Any] = None,
) -> ConformalResult:
    if problem_type == "classification":
        return conformal_classification(model, X_calib, y_calib, X_test, alpha, y_test)
    return conformal_regression(model, X_calib, y_calib, X_test, alpha, y_test)


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
