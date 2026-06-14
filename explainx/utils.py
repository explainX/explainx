"""Model- and data-introspection helpers shared across the engine.

Everything here is model-agnostic: it works with any object exposing the
scikit-learn ``predict`` / ``predict_proba`` convention, which covers
scikit-learn, XGBoost, LightGBM, CatBoost and most hand-rolled estimators.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd


def model_name(model: Any) -> str:
    return type(model).__name__


def detect_problem_type(model: Any, y: Optional[Any] = None) -> str:
    """Return ``"classification"`` or ``"regression"``.

    Prefers scikit-learn's own introspection, then falls back to the presence
    of ``predict_proba`` / ``classes_``, then to the dtype of ``y``.
    """
    try:
        from sklearn.base import is_classifier, is_regressor

        if is_classifier(model):
            return "classification"
        if is_regressor(model):
            return "regression"
    except Exception:
        pass

    if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
        return "classification"

    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in "OUS":  # object / unicode / string
            return "classification"
        unique = np.unique(y_arr[~pd.isna(y_arr)]) if y_arr.dtype.kind == "f" else np.unique(y_arr)
        if y_arr.dtype.kind in "iu" and len(unique) <= max(20, int(0.05 * len(y_arr))):
            return "classification"
    return "regression"


def as_dataframe(X: Any, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
    """Coerce input features into a DataFrame with stable column names."""
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=feature_names)


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Best-effort ``predict`` that tolerates frame/array-only estimators."""
    try:
        return np.asarray(model.predict(X))
    except Exception:
        return np.asarray(model.predict(X.to_numpy()))


def predict_scores(model: Any, X: pd.DataFrame, problem_type: str) -> Optional[np.ndarray]:
    """Continuous score per row used for attribution.

    Classification -> probability of the positive/predicted class.
    Regression     -> the predicted value.
    Returns ``None`` if no usable score is available.
    """
    if problem_type == "regression":
        return predict(model, X).astype(float)

    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
        except Exception:
            proba = np.asarray(model.predict_proba(X.to_numpy()))
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1].astype(float)  # positive class
        if proba.ndim == 2:
            return proba.max(axis=1).astype(float)  # confidence in predicted class
        return proba.astype(float)

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X)).astype(float)
            return scores if scores.ndim == 1 else scores.max(axis=1)
        except Exception:
            return None
    return None


def positive_class(model: Any, y: Optional[Any] = None) -> Any:
    """The class treated as the 'positive' outcome for fairness/scoring."""
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > 0:
        return classes[-1]
    if y is not None:
        return np.unique(np.asarray(y))[-1]
    return 1


def baseline_row(X: pd.DataFrame) -> dict:
    """Per-feature baseline value: median for numerics, mode for the rest."""
    baseline: dict[str, Any] = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            baseline[col] = float(series.median())
        else:
            mode = series.mode()
            baseline[col] = mode.iloc[0] if len(mode) else series.iloc[0]
    return baseline
