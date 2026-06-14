"""Typed, JSON-serializable result objects.

Every explanation the engine produces is one of these dataclasses. They are
deliberately plain data (no model references, no numpy types after
``to_dict``) so the results travel cleanly across an MCP boundary or into an
LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import math


def _clean(value: Any) -> Any:
    """Recursively coerce numpy/pandas scalars to JSON-native types."""
    # Local import keeps the module importable even without numpy at import time.
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, float):
        # JSON has no NaN/Infinity; represent them as None for safe transport.
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


class _Serializable:
    """Mixin adding ``to_dict`` / ``to_json`` to a dataclass."""

    def to_dict(self) -> dict:
        return _clean(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class FeatureImportance(_Serializable):
    """A single feature's global importance score."""

    feature: str
    importance: float
    std: Optional[float] = None


@dataclass
class GlobalImportance(_Serializable):
    """How much each feature matters to the model overall."""

    method: str
    features: list[FeatureImportance] = field(default_factory=list)

    def top(self, n: int = 5) -> list[FeatureImportance]:
        return self.features[:n]


@dataclass
class FeatureContribution(_Serializable):
    """One feature's push on a single prediction."""

    feature: str
    value: Any
    contribution: float  # signed: +pushes prediction up, -pushes it down


@dataclass
class LocalExplanation(_Serializable):
    """Why the model made one specific prediction."""

    index: int
    prediction: Any
    predicted_probability: Optional[float] = None
    baseline: Optional[float] = None
    method: str = "ablation"
    contributions: list[FeatureContribution] = field(default_factory=list)


@dataclass
class GroupFairness(_Serializable):
    """Outcome statistics for one value of a sensitive attribute."""

    group: Any
    count: int
    selection_rate: Optional[float] = None  # P(predicted positive)
    accuracy: Optional[float] = None
    true_positive_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    mean_prediction: Optional[float] = None  # regression


@dataclass
class FairnessReport(_Serializable):
    """Bias analysis across the values of a sensitive attribute."""

    sensitive_feature: str
    problem_type: str
    groups: list[GroupFairness] = field(default_factory=list)
    demographic_parity_difference: Optional[float] = None
    disparate_impact_ratio: Optional[float] = None  # min/max selection rate
    equal_opportunity_difference: Optional[float] = None  # max TPR gap
    biased: bool = False
    findings: list[str] = field(default_factory=list)


@dataclass
class FeatureChange(_Serializable):
    """One feature edit proposed by a counterfactual."""

    feature: str
    original_value: Any
    counterfactual_value: Any


@dataclass
class Counterfactual(_Serializable):
    """The smallest set of changes that flips the model's decision.

    Answers: *"what would have to be different for this prediction to change?"* --
    the most actionable form of explanation for an end user or an LLM debugging
    a model.
    """

    index: int
    original_prediction: Any
    desired_prediction: Any
    found: bool
    changes: list[FeatureChange] = field(default_factory=list)
    new_prediction: Optional[Any] = None
    n_features_changed: int = 0


@dataclass
class PartialDependence(_Serializable):
    """How the model's average output varies with one feature."""

    feature: str
    grid_values: list[float] = field(default_factory=list)
    average_prediction: list[float] = field(default_factory=list)


@dataclass
class ModelMetrics(_Serializable):
    """Standard performance metrics for the model."""

    problem_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[list[list[int]]] = None
    labels: Optional[list[Any]] = None


@dataclass
class ExplanationReport(_Serializable):
    """The full explainability report for a model + dataset."""

    model_name: str
    problem_type: str
    n_samples: int
    n_features: int
    feature_names: list[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    global_importance: Optional[GlobalImportance] = None
    local_explanations: list[LocalExplanation] = field(default_factory=list)
    fairness: list[FairnessReport] = field(default_factory=list)
    summary: str = ""
