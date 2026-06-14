"""LIME: Local Interpretable Model-agnostic Explanations.

LIME explains a single prediction by fitting a simple, interpretable model
(here a weighted linear model) to the black box's behaviour in the *local
neighbourhood* of the instance. We sample perturbations around the instance,
label them with the model, weight them by proximity, and read the linear
coefficients as local contributions.

This is a from-scratch, dependency-light implementation (no `lime` package
required) that returns the same ``LocalExplanation`` shape as the SHAP and
ablation methods, so all three are interchangeable.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import LocalExplanation, FeatureContribution
from .utils import predict, predict_scores


def lime_explain(
    model: Any,
    X: pd.DataFrame,
    index: int,
    problem_type: str,
    n_samples: int = 1000,
    top_k: Optional[int] = None,
    random_state: int = 0,
) -> LocalExplanation:
    from sklearn.linear_model import Ridge

    rng = np.random.RandomState(random_state)
    instance = X.iloc[[index]].copy()

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    means = X[numeric_cols].mean()
    stds = X[numeric_cols].std().replace(0, 1.0)

    # Sample a local neighbourhood: jitter numerics; resample categoricals from data.
    samples = pd.concat([instance] * n_samples, ignore_index=True)
    if numeric_cols:
        noise = rng.normal(0, 1, size=(n_samples, len(numeric_cols))) * stds.to_numpy()
        samples[numeric_cols] = instance[numeric_cols].to_numpy() + noise
    for c in X.columns:
        if c not in numeric_cols:
            samples[c] = rng.choice(X[c].to_numpy(), size=n_samples)

    scores = predict_scores(model, samples, problem_type)
    if scores is None:
        scores = predict(model, samples).astype(float)

    # Standardize numeric features for comparable coefficients; proximity weights.
    Z = np.zeros((n_samples, len(numeric_cols)))
    for j, c in enumerate(numeric_cols):
        Z[:, j] = (samples[c].to_numpy() - means[c]) / stds[c]
    distances = np.sqrt((Z ** 2).sum(axis=1))
    kernel_width = np.sqrt(len(numeric_cols)) * 0.75 or 1.0
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    ridge = Ridge(alpha=1.0)
    ridge.fit(Z, scores, sample_weight=weights)

    inst_z = np.array([(instance[c].to_numpy()[0] - means[c]) / stds[c] for c in numeric_cols])
    contributions = []
    for j, c in enumerate(numeric_cols):
        contributions.append(
            FeatureContribution(
                feature=c,
                value=_native(instance[c].to_numpy()[0]),
                contribution=float(ridge.coef_[j] * inst_z[j]),
            )
        )
    contributions.sort(key=lambda fc: abs(fc.contribution), reverse=True)
    if top_k is not None:
        contributions = contributions[:top_k]

    prediction = predict(model, instance)[0]
    inst_score = predict_scores(model, instance, problem_type)
    return LocalExplanation(
        index=int(index),
        prediction=_native(prediction),
        predicted_probability=float(inst_score[0]) if inst_score is not None and problem_type == "classification" else None,
        baseline=float(ridge.intercept_),
        method="lime",
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
