"""Feature-interaction detection via Friedman's H-statistic.

Importance scores tell you *which* features matter; the H-statistic tells you
which features matter *together* -- i.e. whether the effect of one feature
depends on another. For a pair (j, k) it measures how much the two-feature
partial dependence departs from the sum of the individual partial dependences,
normalized to [0, 1]. High H means a strong interaction the model relies on.
"""

from __future__ import annotations

from typing import Any, Optional
from itertools import combinations
import numpy as np
import pandas as pd

from .schema import InteractionResult, InteractionPair
from .utils import predict_scores


def _pd_1d(model, X, feature, grid, problem_type):
    out = np.empty(len(grid))
    for i, v in enumerate(grid):
        tmp = X.copy()
        tmp[feature] = v
        scores = predict_scores(model, tmp, problem_type)
        out[i] = float(np.mean(scores)) if scores is not None else 0.0
    return out


def h_statistic(
    model: Any,
    X: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    problem_type: str,
    grid_resolution: int = 8,
    sample: int = 200,
) -> float:
    data = X if len(X) <= sample else X.sample(sample, random_state=0)
    ga = np.unique(np.quantile(data[feature_a].astype(float), np.linspace(0, 1, grid_resolution)))
    gb = np.unique(np.quantile(data[feature_b].astype(float), np.linspace(0, 1, grid_resolution)))

    pd_a = _pd_1d(model, data, feature_a, ga, problem_type)
    pd_b = _pd_1d(model, data, feature_b, gb, problem_type)
    # Center each partial dependence to mean zero (Friedman's convention).
    pda_c = dict(zip(ga, pd_a - pd_a.mean()))
    pdb_c = dict(zip(gb, pd_b - pd_b.mean()))

    # Joint PD over the grid, then center.
    joint = np.empty((len(ga), len(gb)))
    for i, va in enumerate(ga):
        for j, vb in enumerate(gb):
            tmp = data.copy()
            tmp[feature_a] = va
            tmp[feature_b] = vb
            scores = predict_scores(model, tmp, problem_type)
            joint[i, j] = float(np.mean(scores)) if scores is not None else 0.0
    joint_c = joint - joint.mean()

    numerator, denominator = 0.0, 0.0
    for i, va in enumerate(ga):
        for j, vb in enumerate(gb):
            diff = joint_c[i, j] - pda_c[va] - pdb_c[vb]
            numerator += diff ** 2
            denominator += joint_c[i, j] ** 2
    if denominator == 0:
        return 0.0
    return float(np.sqrt(max(0.0, numerator) / denominator))


def feature_interactions(
    model: Any,
    X: pd.DataFrame,
    problem_type: str,
    features: Optional[list[str]] = None,
    top_k: int = 5,
    grid_resolution: int = 8,
) -> InteractionResult:
    numeric = [c for c in (features or X.columns) if pd.api.types.is_numeric_dtype(X[c])]
    pairs: list[InteractionPair] = []
    for a, b in combinations(numeric, 2):
        try:
            strength = h_statistic(model, X, a, b, problem_type, grid_resolution=grid_resolution)
        except Exception:
            continue
        pairs.append(InteractionPair(feature_a=a, feature_b=b, strength=min(1.0, strength)))

    pairs.sort(key=lambda p: p.strength, reverse=True)
    return InteractionResult(method="friedman_h", pairs=pairs[:top_k])
