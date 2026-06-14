"""Accumulated Local Effects (ALE).

ALE describes how a feature influences predictions *on average*, like a partial
dependence plot -- but unlike PDP it stays unbiased when features are
correlated, because it integrates the model's **local** changes within narrow
bins instead of marginalising over the (possibly unrealistic) full distribution.
Apley & Zhu (2020) showed ALE is the more trustworthy global effect curve, and
it is the modern default for correlated tabular data.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .schema import ALEResult
from .utils import predict_scores


def ale(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    problem_type: str,
    n_bins: int = 10,
) -> ALEResult:
    if feature not in X.columns:
        raise ValueError(f"feature '{feature}' not in data columns")
    if not pd.api.types.is_numeric_dtype(X[feature]):
        raise ValueError("ALE currently supports numeric features only.")

    values = X[feature].to_numpy(dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        return ALEResult(feature=feature, bin_edges=[float(edges[0])], ale=[0.0])

    # Assign each row to a bin.
    bin_idx = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)

    local_effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        lo = X[mask].copy()
        hi = X[mask].copy()
        lo[feature] = edges[b]
        hi[feature] = edges[b + 1]
        s_lo = predict_scores(model, lo, problem_type)
        s_hi = predict_scores(model, hi, problem_type)
        if s_lo is None or s_hi is None:
            continue
        local_effects[b] = float(np.mean(np.asarray(s_hi) - np.asarray(s_lo)))
        counts[b] = n

    accumulated = np.cumsum(local_effects)
    # Center so the average effect over the data is zero (standard ALE convention).
    total = counts.sum()
    if total > 0:
        mean_effect = np.sum((accumulated * counts)) / total
        accumulated = accumulated - mean_effect

    return ALEResult(
        feature=feature,
        bin_edges=[float(e) for e in edges],
        ale=[0.0] + [float(a) for a in accumulated],
    )
