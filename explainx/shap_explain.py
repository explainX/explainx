"""Optional SHAP integration -- the state-of-the-art attribution method.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
grounded in cooperative game theory, with strong consistency guarantees that
the simple ablation fallback does not have. We use it automatically *when it is
installed*, and degrade gracefully to the built-in ablation method otherwise,
so SHAP stays an optional dependency.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import GlobalImportance, FeatureImportance, LocalExplanation, FeatureContribution


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def _build_explainer(model: Any, X: pd.DataFrame, max_background: int = 100):
    """Pick the most appropriate SHAP explainer for the model."""
    import shap

    background = X if len(X) <= max_background else X.sample(max_background, random_state=0)
    # shap.Explainer auto-routes to TreeExplainer / LinearExplainer / Kernel etc.
    try:
        return shap.Explainer(model, background)
    except Exception:
        predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        return shap.Explainer(predict, background)


def _values_2d(shap_values, predicted_class: int) -> np.ndarray:
    """Normalize a shap values array to 2-D (n_samples, n_features)."""
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (samples, features, classes) -> pick the predicted/positive class slice.
        idx = min(predicted_class, arr.shape[2] - 1)
        return arr[:, :, idx]
    return arr


def shap_global_importance(
    model: Any, X: pd.DataFrame, max_samples: int = 200
) -> Optional[GlobalImportance]:
    if not shap_available():
        return None
    try:
        import shap  # noqa: F401

        sample = X if len(X) <= max_samples else X.sample(max_samples, random_state=0)
        explainer = _build_explainer(model, X)
        explanation = explainer(sample)
        vals = _values_2d(explanation.values, predicted_class=1)
        mean_abs = np.abs(vals).mean(axis=0)
        feats = [
            FeatureImportance(feature=col, importance=float(mean_abs[i]))
            for i, col in enumerate(X.columns)
        ]
        feats.sort(key=lambda f: abs(f.importance), reverse=True)
        return GlobalImportance(method="shap_mean_abs", features=feats)
    except Exception:
        return None


def shap_local_explanation(
    model: Any, X: pd.DataFrame, index: int, problem_type: str, top_k: Optional[int] = None
) -> Optional[LocalExplanation]:
    if not shap_available():
        return None
    try:
        explainer = _build_explainer(model, X)
        instance = X.iloc[[index]]
        explanation = explainer(instance)

        predicted_class = 1
        if problem_type == "classification" and hasattr(model, "predict"):
            pred = model.predict(instance)[0]
            classes = list(getattr(model, "classes_", []))
            if classes and pred in classes:
                predicted_class = classes.index(pred)

        vals = _values_2d(explanation.values, predicted_class)[0]
        base = explanation.base_values
        base_arr = np.asarray(base)
        if base_arr.ndim >= 1 and base_arr.size > 1:
            baseline = float(base_arr.flatten()[min(predicted_class, base_arr.size - 1)])
        else:
            baseline = float(base_arr) if base_arr.size else None

        contributions = [
            FeatureContribution(
                feature=col,
                value=_native(instance.iloc[0][col]),
                contribution=float(vals[i]),
            )
            for i, col in enumerate(X.columns)
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        if top_k is not None:
            contributions = contributions[:top_k]

        prediction = model.predict(instance)[0]
        return LocalExplanation(
            index=int(index),
            prediction=_native(prediction),
            baseline=baseline,
            method="shap",
            contributions=contributions,
        )
    except Exception:
        return None


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
