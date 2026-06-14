"""Global feature importance: which features drive the model overall.

Strategy, in order of preference:
1. Permutation importance (model-agnostic, ground-truth aware) when ``y`` is
   available -- this is the most trustworthy and works for any estimator.
2. The model's own ``feature_importances_`` (tree ensembles).
3. The magnitude of linear ``coef_`` (linear / logistic models).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import GlobalImportance, FeatureImportance


def global_importance(
    model: Any,
    X: pd.DataFrame,
    y: Optional[Any] = None,
    problem_type: str = "classification",
    n_repeats: int = 10,
    random_state: int = 0,
    use_shap: bool = True,
) -> GlobalImportance:
    feature_names = list(X.columns)

    if use_shap:
        from .shap_explain import shap_global_importance

        shap_result = shap_global_importance(model, X)
        if shap_result is not None:
            return shap_result

    if y is not None:
        try:
            return _permutation_importance(
                model, X, y, problem_type, feature_names, n_repeats, random_state
            )
        except Exception:
            pass  # fall through to model-intrinsic importance

    intrinsic = _intrinsic_importance(model, feature_names)
    if intrinsic is not None:
        return intrinsic

    # Last resort: empty/zero importances so callers still get a valid object.
    return GlobalImportance(
        method="unavailable",
        features=[FeatureImportance(f, 0.0) for f in feature_names],
    )


def _permutation_importance(
    model, X, y, problem_type, feature_names, n_repeats, random_state
) -> GlobalImportance:
    from sklearn.inspection import permutation_importance

    scoring = "r2" if problem_type == "regression" else "accuracy"
    result = permutation_importance(
        model, X, np.asarray(y),
        n_repeats=n_repeats, random_state=random_state, scoring=scoring,
    )
    feats = [
        FeatureImportance(
            feature=name,
            importance=float(result.importances_mean[i]),
            std=float(result.importances_std[i]),
        )
        for i, name in enumerate(feature_names)
    ]
    feats.sort(key=lambda f: abs(f.importance), reverse=True)
    return GlobalImportance(method="permutation_importance", features=feats)


def _intrinsic_importance(model, feature_names) -> Optional[GlobalImportance]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "coef_magnitude"
    else:
        return None

    if len(values) != len(feature_names):
        return None

    feats = [FeatureImportance(name, float(values[i])) for i, name in enumerate(feature_names)]
    feats.sort(key=lambda f: abs(f.importance), reverse=True)
    return GlobalImportance(method=method, features=feats)
