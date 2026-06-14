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
class SurrogateExplanation(_Serializable):
    """A glassbox model trained to mimic the black box (global surrogate)."""

    method: str
    fidelity: float  # how well the surrogate agrees with the model on the data
    fidelity_metric: str  # "accuracy" (classification) or "r2" (regression)
    max_depth: int
    rules_text: str = ""
    feature_importances: list[FeatureImportance] = field(default_factory=list)


@dataclass
class ALEResult(_Serializable):
    """Accumulated Local Effects: feature effect that is unbiased under correlation."""

    feature: str
    bin_edges: list[float] = field(default_factory=list)
    ale: list[float] = field(default_factory=list)  # centered effect per bin


@dataclass
class Anchor(_Serializable):
    """A high-precision IF-THEN rule that 'anchors' a prediction locally."""

    index: int
    prediction: Any
    rules: list[str] = field(default_factory=list)
    precision: float = 0.0  # P(same prediction | rule holds)
    coverage: float = 0.0  # fraction of data the rule applies to


@dataclass
class ExplanationQuality(_Serializable):
    """Quantified trustworthiness of an explanation (2024-2025 XAI eval research)."""

    method: str
    faithfulness: Optional[float] = None  # do attributions match feature-removal effects?
    stability: Optional[float] = None  # are attributions stable under small input changes?
    notes: list[str] = field(default_factory=list)


@dataclass
class DriftFeature(_Serializable):
    feature: str
    psi: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    drifted: bool = False


@dataclass
class DriftReport(_Serializable):
    """Distribution shift between a reference and a current dataset."""

    n_features: int
    n_drifted: int
    psi_threshold: float
    drifted: bool
    features: list[DriftFeature] = field(default_factory=list)


@dataclass
class ConformalResult(_Serializable):
    """Distribution-free uncertainty via split conformal prediction.

    Gives a finite-sample coverage guarantee: prediction sets (classification)
    or intervals (regression) that contain the truth with probability >= 1-alpha,
    with no assumptions about the model.
    """

    problem_type: str
    alpha: float
    coverage_target: float  # 1 - alpha
    qhat: float
    prediction_sets: Optional[list[list[Any]]] = None  # classification
    intervals: Optional[list[list[float]]] = None  # regression [lo, hi]
    average_set_size: Optional[float] = None
    average_interval_width: Optional[float] = None
    empirical_coverage: Optional[float] = None  # if labels supplied


@dataclass
class MitigationThreshold(_Serializable):
    group: Any
    threshold: float
    selection_rate: float


@dataclass
class MitigationResult(_Serializable):
    """Post-processing bias mitigation via per-group decision thresholds."""

    sensitive_feature: str
    objective: str  # "demographic_parity"
    thresholds: list[MitigationThreshold] = field(default_factory=list)
    parity_gap_before: Optional[float] = None
    parity_gap_after: Optional[float] = None
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class InteractionPair(_Serializable):
    feature_a: str
    feature_b: str
    strength: float  # Friedman H-statistic in [0, 1]


@dataclass
class InteractionResult(_Serializable):
    """Pairwise feature-interaction strengths (Friedman's H-statistic)."""

    method: str
    pairs: list[InteractionPair] = field(default_factory=list)


@dataclass
class PrototypesResult(_Serializable):
    """Example-based explanation: representative prototypes + atypical criticisms."""

    method: str
    prototype_indices: list[int] = field(default_factory=list)
    criticism_indices: list[int] = field(default_factory=list)


@dataclass
class ErrorSlice(_Serializable):
    feature: str
    condition: str
    size: int
    error: float
    baseline_error: float
    lift: float  # slice error / baseline error (>1 means worse than average)


@dataclass
class ErrorAnalysis(_Serializable):
    """Where the model fails: data slices with elevated error (slice discovery)."""

    problem_type: str
    metric: str
    baseline_error: float
    slices: list[ErrorSlice] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class LabelIssue(_Serializable):
    index: int
    given_label: Any
    suggested_label: Any
    confidence: float


@dataclass
class LabelIssues(_Serializable):
    """Likely-mislabeled rows (confident-learning style) to review/relabel."""

    n_checked: int
    n_issues: int
    estimated_noise_rate: float
    out_of_sample: bool
    issues: list[LabelIssue] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class LeakageFeature(_Serializable):
    feature: str
    solo_score: float
    suspected: bool


@dataclass
class LeakageReport(_Serializable):
    """Features that alone predict the target suspiciously well (target leakage)."""

    metric: str
    threshold: float
    features: list[LeakageFeature] = field(default_factory=list)
    suspected_leakage: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class CalibrationBin(_Serializable):
    mean_confidence: float
    accuracy: float
    count: int


@dataclass
class CalibrationReport(_Serializable):
    """How well predicted probabilities match observed frequencies."""

    ece: float
    brier: float
    well_calibrated: bool
    bins: list[CalibrationBin] = field(default_factory=list)
    recommendation: str = ""


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
    surrogate: Optional[SurrogateExplanation] = None
    explanation_quality: Optional[ExplanationQuality] = None
    summary: str = ""
