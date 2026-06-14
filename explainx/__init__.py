"""explainX-LLM: an LLM-native model explainability engine.

This package is a modern, dependency-light rewrite of the explainX idea, built
so that *both* humans and LLM agents can inspect a trained machine-learning
model, understand why it makes a prediction, surface bias, and feed those
insights back into training.

Unlike the legacy explainX (a Plotly Dash dashboard meant for human eyes), this
core returns **structured, machine-readable results** (typed objects that
serialize to JSON) plus a natural-language summary. That makes it usable from
plain Python *and* over the Model Context Protocol (see ``explainx.mcp_server``).

Quick start
-----------
>>> from explainx import explain_model
>>> report = explain_model(model, X_test, y_test)
>>> print(report.summary)          # natural-language overview for a human/LLM
>>> report.to_dict()               # structured JSON-ready dict
"""

from .explainer import ModelExplainer, explain_model
from .schema import (
    ExplanationReport,
    FeatureImportance,
    GlobalImportance,
    LocalExplanation,
    FeatureContribution,
    FairnessReport,
    GroupFairness,
    ModelMetrics,
    Counterfactual,
    FeatureChange,
    PartialDependence,
    SurrogateExplanation,
    ALEResult,
    Anchor,
    ExplanationQuality,
    DriftReport,
    DriftFeature,
    ConformalResult,
    MitigationResult,
    InteractionResult,
    InteractionPair,
    PrototypesResult,
)
from .drift import detect_drift
from .report_html import report_to_html, save_html
from .conformal import conformal_predict
from .prototypes import prototypes_and_criticisms
from .adapters import wrap_model
from .diagnostics import error_analysis, find_label_issues, detect_leakage, assess_calibration

__all__ = [
    "explain_model",
    "ModelExplainer",
    "ExplanationReport",
    "FeatureImportance",
    "GlobalImportance",
    "LocalExplanation",
    "FeatureContribution",
    "FairnessReport",
    "GroupFairness",
    "ModelMetrics",
    "Counterfactual",
    "FeatureChange",
    "PartialDependence",
    "SurrogateExplanation",
    "ALEResult",
    "Anchor",
    "ExplanationQuality",
    "DriftReport",
    "DriftFeature",
    "detect_drift",
    "report_to_html",
    "save_html",
    "ConformalResult",
    "MitigationResult",
    "InteractionResult",
    "InteractionPair",
    "PrototypesResult",
    "conformal_predict",
    "prototypes_and_criticisms",
    "wrap_model",
    "error_analysis",
    "find_label_issues",
    "detect_leakage",
    "assess_calibration",
]

__version__ = "3.0.0"
