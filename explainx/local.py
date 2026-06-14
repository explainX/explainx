"""Local explanations: why the model made one specific prediction.

We use a model-agnostic *ablation* (occlusion) attribution. For an instance
``x`` with model score ``f(x)``, each feature ``j`` is reset to its dataset
baseline (median for numerics, mode otherwise) to produce ``x_j``. The signed
contribution of feature ``j`` is ``f(x) - f(x_j)``:

* a positive contribution means the feature's actual value pushed the
  prediction *up* (toward the positive class / a larger value),
* a negative contribution means it pushed the prediction *down*.

This needs no extra dependencies and works for any predict/predict_proba
estimator. It is an approximation (it ignores feature interactions), which the
output labels make explicit. If SHAP is installed it is used instead for
sharper, interaction-aware attributions.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import LocalExplanation, FeatureContribution
from .utils import predict, predict_scores, baseline_row


def explain_instance(
    model: Any,
    X: pd.DataFrame,
    index: int,
    problem_type: str,
    background: Optional[pd.DataFrame] = None,
    top_k: Optional[int] = None,
    use_shap: bool = True,
) -> LocalExplanation:
    if use_shap:
        from .shap_explain import shap_local_explanation

        shap_result = shap_local_explanation(model, X, index, problem_type, top_k=top_k)
        if shap_result is not None:
            return shap_result

    background = X if background is None else background
    baseline = baseline_row(background)

    instance = X.iloc[[index]].copy()
    score = predict_scores(model, instance, problem_type)
    instance_score = float(score[0]) if score is not None else None

    baseline_frame = instance.copy()
    for col, val in baseline.items():
        baseline_frame[col] = val
    base_score_arr = predict_scores(model, baseline_frame, problem_type)
    base_score = float(base_score_arr[0]) if base_score_arr is not None else None

    contributions: list[FeatureContribution] = []
    for col in X.columns:
        if baseline[col] == instance.iloc[0][col]:
            contribution = 0.0
        else:
            perturbed = instance.copy()
            perturbed[col] = baseline[col]
            perturbed_score = predict_scores(model, perturbed, problem_type)
            if instance_score is None or perturbed_score is None:
                contribution = 0.0
            else:
                contribution = instance_score - float(perturbed_score[0])
        contributions.append(
            FeatureContribution(
                feature=col,
                value=_native(instance.iloc[0][col]),
                contribution=float(contribution),
            )
        )

    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    if top_k is not None:
        contributions = contributions[:top_k]

    prediction = predict(model, instance)[0]
    return LocalExplanation(
        index=int(index),
        prediction=_native(prediction),
        predicted_probability=instance_score if problem_type == "classification" else None,
        baseline=base_score,
        method="ablation",
        contributions=contributions,
    )


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
