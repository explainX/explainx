"""Partial dependence: the marginal effect of a feature on the prediction.

A PDP shows how the model's *average* output changes as one feature sweeps
across its range, holding the data distribution of the others fixed. It is the
standard tool for reading the shape of a learned relationship (monotonic?
threshold? non-linear?) and for sanity-checking that the model behaves the way
a domain expert expects.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .schema import PartialDependence


def partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 20,
) -> PartialDependence:
    if feature not in X.columns:
        raise ValueError(f"feature '{feature}' not in data columns")

    try:
        from sklearn.inspection import partial_dependence as sk_pd

        result = sk_pd(model, X, [list(X.columns).index(feature)], grid_resolution=grid_resolution)
        grid = np.asarray(result["grid_values"][0], dtype=float)
        avg = np.asarray(result["average"][0], dtype=float)
        return PartialDependence(
            feature=feature,
            grid_values=[float(g) for g in grid],
            average_prediction=[float(a) for a in avg],
        )
    except Exception:
        return _manual_pdp(model, X, feature, grid_resolution)


def _manual_pdp(model, X, feature, grid_resolution) -> PartialDependence:
    """Model-agnostic fallback that works for any estimator and dtype."""
    from .utils import predict_scores, detect_problem_type

    problem_type = detect_problem_type(model)
    series = X[feature]
    if pd.api.types.is_numeric_dtype(series):
        grid = np.linspace(series.min(), series.max(), grid_resolution)
    else:
        grid = pd.unique(series)

    averages = []
    for value in grid:
        perturbed = X.copy()
        perturbed[feature] = value
        scores = predict_scores(model, perturbed, problem_type)
        averages.append(float(np.mean(scores)) if scores is not None else float("nan"))

    return PartialDependence(
        feature=feature,
        grid_values=[float(g) if isinstance(g, (int, float, np.floating, np.integer)) else g for g in grid],
        average_prediction=averages,
    )
