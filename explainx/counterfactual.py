"""Counterfactual explanations: the minimal change that flips a decision.

Given an instance the model classified as, say, "rejected", a counterfactual
answers *"what is the smallest change to the inputs that would have produced
'approved' instead?"*. This is the most actionable explanation type -- it tells
a user (or an LLM repairing a model) exactly which levers matter at the margin.

The search is model-agnostic and greedy: at each step it tries moving each
not-yet-changed feature toward candidate values drawn from the background data
and keeps the single edit that most increases the desired-class score, until
the prediction flips or a change budget is exhausted.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import Counterfactual, FeatureChange
from .utils import predict, predict_scores, positive_class


def _candidate_values(series: pd.Series, n: int = 9) -> list:
    if pd.api.types.is_numeric_dtype(series):
        qs = np.linspace(0.0, 1.0, n)
        return sorted(set(float(series.quantile(q)) for q in qs))
    return list(pd.unique(series))


def find_counterfactual(
    model: Any,
    X: pd.DataFrame,
    index: int,
    problem_type: str,
    desired_class: Any = None,
    max_changes: int = 4,
    background: Optional[pd.DataFrame] = None,
    immutable_features: Optional[list] = None,
    feature_directions: Optional[dict] = None,
) -> Counterfactual:
    """Find the smallest change that flips a prediction.

    For *actionable recourse*, pass ``immutable_features`` (columns that can
    never change, e.g. ``age``, ``gender``) and/or ``feature_directions`` mapping
    a column to ``"increase"`` or ``"decrease"`` (e.g. income can only go up).
    The search then proposes only realistic, permitted edits.
    """
    if problem_type != "classification":
        raise ValueError("Counterfactuals are currently supported for classification only.")

    immutable = set(immutable_features or [])
    directions = feature_directions or {}
    background = X if background is None else background
    instance = X.iloc[[index]].copy()
    original_pred = predict(model, instance)[0]

    if desired_class is None:
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2:
            desired_class = classes[0] if classes[1] == original_pred else classes[1]
        else:
            raise ValueError("desired_class is required for multiclass counterfactuals.")

    pos = positive_class(model)
    # Direction of the score we want to push: toward desired_class.
    want_high = desired_class == pos

    candidates = {col: _candidate_values(background[col]) for col in X.columns}
    current = instance.copy()
    changed: dict[str, Any] = {}

    def desired_score(frame: pd.DataFrame) -> float:
        score = predict_scores(model, frame, "classification")
        s = float(score[0]) if score is not None else 0.0
        return s if want_high else -s

    for _ in range(max_changes):
        if predict(model, current)[0] == desired_class:
            break
        best_gain, best_col, best_val = -np.inf, None, None
        base_score = desired_score(current)
        for col in X.columns:
            if col in changed or col in immutable:
                continue
            current_val = current.iloc[0][col]
            for val in candidates[col]:
                if val == current_val:
                    continue
                # Respect monotonic recourse constraints when provided.
                direction = directions.get(col)
                if direction == "increase" and not val > current_val:
                    continue
                if direction == "decrease" and not val < current_val:
                    continue
                trial = current.copy()
                trial[col] = val
                gain = desired_score(trial) - base_score
                if gain > best_gain:
                    best_gain, best_col, best_val = gain, col, val
        if best_col is None or best_gain <= 0:
            break
        current[best_col] = best_val
        changed[best_col] = best_val

    new_pred = predict(model, current)[0]
    found = new_pred == desired_class
    changes = [
        FeatureChange(
            feature=col,
            original_value=_native(instance.iloc[0][col]),
            counterfactual_value=_native(val),
        )
        for col, val in changed.items()
    ]
    return Counterfactual(
        index=int(index),
        original_prediction=_native(original_pred),
        desired_prediction=_native(desired_class),
        found=bool(found),
        changes=changes,
        new_prediction=_native(new_pred),
        n_features_changed=len(changes),
    )


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
