"""Bias mitigation: move from *detecting* unfairness to *fixing* it.

Detection (``fairness.py``) tells you a model is biased; this closes the loop by
computing a remedy. We use **post-processing**: pick a per-group decision
threshold on the model's score so that every group's selection rate matches a
common target, equalizing demographic parity without retraining. This is the
fastest, model-agnostic mitigation and directly supports the vision's "go back
and fix" step -- an LLM agent can read the recommended thresholds and apply them.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import MitigationResult, MitigationThreshold
from .utils import predict_scores, positive_class


def mitigate_demographic_parity(
    model: Any,
    X: pd.DataFrame,
    sensitive_feature: str,
    y_true: Optional[Any] = None,
    target_rate: Optional[float] = None,
) -> MitigationResult:
    if sensitive_feature not in X.columns:
        raise ValueError(f"sensitive_feature '{sensitive_feature}' not in data columns")

    scores = predict_scores(model, X, "classification")
    if scores is None:
        raise ValueError("Mitigation needs a model with probability/decision scores.")
    scores = np.asarray(scores)
    groups = X[sensitive_feature]
    pos = positive_class(model, y_true)

    # Default target: the overall positive rate at the model's natural 0.5 cut.
    if target_rate is None:
        target_rate = float(np.mean(scores >= 0.5))

    result = MitigationResult(
        sensitive_feature=sensitive_feature, objective="demographic_parity"
    )

    baseline_pred = (scores >= 0.5).astype(int)
    mitigated_pred = baseline_pred.copy()

    base_rates = {}
    for group in pd.unique(groups):
        mask = (groups == group).to_numpy()
        gscores = scores[mask]
        base_rates[group] = float(np.mean(gscores >= 0.5))

        # Threshold whose selection rate is closest to the target for this group.
        candidate_thresholds = np.unique(np.quantile(gscores, np.linspace(0, 1, 101)))
        best_t, best_diff = 0.5, np.inf
        for t in candidate_thresholds:
            rate = float(np.mean(gscores >= t))
            diff = abs(rate - target_rate)
            if diff < best_diff:
                best_diff, best_t = diff, t
        achieved = float(np.mean(gscores >= best_t))
        result.thresholds.append(
            MitigationThreshold(group=_native(group), threshold=float(best_t), selection_rate=achieved)
        )
        mitigated_pred[mask] = (gscores >= best_t).astype(int)

    rates_before = list(base_rates.values())
    rates_after = [t.selection_rate for t in result.thresholds]
    result.parity_gap_before = float(max(rates_before) - min(rates_before)) if rates_before else None
    result.parity_gap_after = float(max(rates_after) - min(rates_after)) if rates_after else None

    if y_true is not None:
        y_true = np.asarray(y_true)
        truth_pos = (y_true == pos).astype(int)
        result.accuracy_before = float(np.mean(baseline_pred == truth_pos))
        result.accuracy_after = float(np.mean(mitigated_pred == truth_pos))

    result.notes.append(
        f"Apply per-group thresholds to equalize selection rate near {target_rate:.1%}. "
        f"Parity gap {result.parity_gap_before:.1%} -> {result.parity_gap_after:.1%}."
        if result.parity_gap_before is not None else "Per-group thresholds computed."
    )
    return result


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
