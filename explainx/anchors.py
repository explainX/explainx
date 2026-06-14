"""Anchors: high-precision IF-THEN rules that pin down a prediction.

An anchor is a small set of conditions such that, *whenever they hold*, the
model almost always returns the same prediction -- e.g. "IF credit_score > 700
AND debt_ratio <= 0.3 THEN approved (precision 0.97)". Anchors (Ribeiro et al.,
2018) complement attributions: instead of "how much did each feature matter",
they answer "what is sufficient to guarantee this outcome".

This is a pragmatic, model-agnostic implementation: candidate predicates are
quantile bins for numerics and equality for categoricals, added greedily to
maximise precision (estimated by perturbing the non-anchored features) until a
precision threshold is met.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import Anchor
from .utils import predict


def _candidate_predicates(X: pd.DataFrame, instance: pd.Series, n_bins: int = 5):
    """Yield (column, description, mask_fn) predicates satisfied by the instance."""
    preds = []
    for col in X.columns:
        val = instance[col]
        if pd.api.types.is_numeric_dtype(X[col]):
            edges = np.unique(np.quantile(X[col].to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.searchsorted(edges, val, side="left") - 1, 0, len(edges) - 2)
            lo, hi = edges[b], edges[b + 1]
            desc = f"{lo:.3g} <= {col} <= {hi:.3g}"
            preds.append((col, desc, lambda df, lo=lo, hi=hi, c=col: (df[c] >= lo) & (df[c] <= hi)))
        else:
            desc = f"{col} == {val}"
            preds.append((col, desc, lambda df, v=val, c=col: df[c] == v))
    return preds


def find_anchor(
    model: Any,
    X: pd.DataFrame,
    index: int,
    precision_threshold: float = 0.95,
    max_predicates: int = 4,
    n_samples: int = 2000,
    random_state: int = 0,
) -> Anchor:
    rng = np.random.RandomState(random_state)
    instance = X.iloc[index]
    target_pred = predict(model, X.iloc[[index]])[0]

    # Background sample we perturb to estimate rule precision/coverage.
    bg = X.sample(min(n_samples, len(X)), replace=len(X) < n_samples, random_state=random_state).reset_index(drop=True)
    bg_pred = predict(model, bg)

    candidates = _candidate_predicates(X, instance)
    chosen: list = []
    chosen_cols: set = set()

    def precision_coverage(rules):
        mask = np.ones(len(bg), dtype=bool)
        for _, _, fn in rules:
            mask &= fn(bg).to_numpy()
        cov = float(mask.mean())
        if mask.sum() == 0:
            return 0.0, 0.0
        prec = float(np.mean(bg_pred[mask] == target_pred))
        return prec, cov

    best_prec, best_cov = 0.0, 1.0
    for _ in range(max_predicates):
        best_gain, best_pred = -np.inf, None
        for cand in candidates:
            if cand[0] in chosen_cols:
                continue
            prec, cov = precision_coverage(chosen + [cand])
            # Prefer precision; break ties by coverage.
            gain = prec + 0.01 * cov
            if gain > best_gain:
                best_gain, best_pred, best_prec, best_cov = gain, cand, prec, cov
        if best_pred is None:
            break
        chosen.append(best_pred)
        chosen_cols.add(best_pred[0])
        if best_prec >= precision_threshold:
            break

    prec, cov = precision_coverage(chosen)
    return Anchor(
        index=int(index),
        prediction=_native(target_pred),
        rules=[desc for _, desc, _ in chosen],
        precision=float(prec),
        coverage=float(cov),
    )


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
