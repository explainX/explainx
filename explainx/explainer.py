"""The orchestrator: tie metrics, importance, local and fairness together.

``explain_model`` is the one-call entry point that produces a full
``ExplanationReport``. ``ModelExplainer`` is the stateful equivalent for when
you want to run individual analyses on demand (handy for an agent that probes
the model iteratively).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from .schema import ExplanationReport
from .utils import as_dataframe, detect_problem_type, model_name
from .metrics import compute_metrics
from .importance import global_importance
from .local import explain_instance
from .fairness import fairness_analysis
from .counterfactual import find_counterfactual
from .pdp import partial_dependence
from .surrogate import global_surrogate
from .lime_explain import lime_explain
from .anchors import find_anchor
from .ale import ale as _ale
from .quality import evaluate_local
from .conformal import conformal_predict
from .mitigation import mitigate_demographic_parity
from .interactions import feature_interactions
from .prototypes import prototypes_and_criticisms
from .schema import PrototypesResult
from .summary import build_summary


class ModelExplainer:
    """Inspect a trained model on a dataset, one analysis at a time."""

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        feature_names: Optional[list[str]] = None,
        problem_type: Optional[str] = None,
        use_shap: bool = True,
    ):
        self.model = model
        self.X = as_dataframe(X, feature_names)
        self.y = None if y is None else np.asarray(y)
        self.problem_type = problem_type or detect_problem_type(model, self.y)
        self.use_shap = use_shap

    # --- individual analyses ------------------------------------------------
    def metrics(self):
        if self.y is None:
            raise ValueError("Ground-truth y is required to compute metrics.")
        return compute_metrics(self.model, self.X, self.y, self.problem_type)

    def importance(self, n_repeats: int = 10):
        return global_importance(
            self.model, self.X, self.y, self.problem_type,
            n_repeats=n_repeats, use_shap=self.use_shap,
        )

    def explain(self, index: int, top_k: Optional[int] = None):
        return explain_instance(
            self.model, self.X, index, self.problem_type,
            top_k=top_k, use_shap=self.use_shap,
        )

    def fairness(self, sensitive_feature: str, pos_label: Any = None):
        return fairness_analysis(
            self.model, self.X, sensitive_feature, self.y, self.problem_type, pos_label
        )

    def counterfactual(self, index: int, desired_class: Any = None, max_changes: int = 4):
        return find_counterfactual(
            self.model, self.X, index, self.problem_type,
            desired_class=desired_class, max_changes=max_changes,
        )

    def partial_dependence(self, feature: str, grid_resolution: int = 20):
        return partial_dependence(self.model, self.X, feature, grid_resolution=grid_resolution)

    def surrogate(self, max_depth: int = 4):
        return global_surrogate(self.model, self.X, self.problem_type, max_depth=max_depth)

    def lime(self, index: int, top_k: Optional[int] = None, n_samples: int = 1000):
        return lime_explain(
            self.model, self.X, index, self.problem_type, n_samples=n_samples, top_k=top_k
        )

    def anchor(self, index: int, precision_threshold: float = 0.95, max_predicates: int = 4):
        return find_anchor(
            self.model, self.X, index,
            precision_threshold=precision_threshold, max_predicates=max_predicates,
        )

    def ale(self, feature: str, n_bins: int = 10):
        return _ale(self.model, self.X, feature, self.problem_type, n_bins=n_bins)

    def explanation_quality(self, index: int = 0, top_k: Optional[int] = None):
        local = self.explain(index, top_k=top_k)
        return evaluate_local(self.model, self.X, local, self.problem_type)

    def recourse(
        self,
        index: int,
        desired_class: Any = None,
        immutable_features: Optional[list] = None,
        feature_directions: Optional[dict] = None,
        max_changes: int = 5,
    ):
        return find_counterfactual(
            self.model, self.X, index, self.problem_type,
            desired_class=desired_class, max_changes=max_changes,
            immutable_features=immutable_features, feature_directions=feature_directions,
        )

    def conformal(
        self,
        X_calib: Any,
        y_calib: Any,
        X_test: Optional[Any] = None,
        alpha: float = 0.1,
        y_test: Optional[Any] = None,
    ):
        X_calib = as_dataframe(X_calib, list(self.X.columns))
        X_test = self.X if X_test is None else as_dataframe(X_test, list(self.X.columns))
        return conformal_predict(
            self.model, X_calib, y_calib, X_test, self.problem_type, alpha=alpha, y_test=y_test
        )

    def mitigate_bias(self, sensitive_feature: str, target_rate: Optional[float] = None):
        return mitigate_demographic_parity(
            self.model, self.X, sensitive_feature, self.y, target_rate=target_rate
        )

    def interactions(self, top_k: int = 5, features: Optional[Sequence[str]] = None):
        return feature_interactions(
            self.model, self.X, self.problem_type,
            features=list(features) if features else None, top_k=top_k,
        )

    def prototypes(self, n_prototypes: int = 5, n_criticisms: int = 3):
        protos, crits = prototypes_and_criticisms(self.X, n_prototypes, n_criticisms)
        return PrototypesResult(
            method="mmd_rbf", prototype_indices=protos, criticism_indices=crits
        )

    # --- full report --------------------------------------------------------
    def report(
        self,
        sensitive_features: Optional[Sequence[str]] = None,
        explain_indices: Optional[Sequence[int]] = None,
        n_local: int = 3,
        top_k: Optional[int] = 10,
        include_surrogate: bool = True,
        include_quality: bool = True,
    ) -> ExplanationReport:
        report = ExplanationReport(
            model_name=model_name(self.model),
            problem_type=self.problem_type,
            n_samples=int(len(self.X)),
            n_features=int(self.X.shape[1]),
            feature_names=list(self.X.columns),
        )

        if self.y is not None:
            try:
                report.metrics = self.metrics()
            except Exception:
                pass

        try:
            report.global_importance = self.importance()
        except Exception:
            pass

        if explain_indices is None:
            explain_indices = list(range(min(n_local, len(self.X))))
        for idx in explain_indices:
            try:
                report.local_explanations.append(self.explain(int(idx), top_k=top_k))
            except Exception:
                pass

        for feat in sensitive_features or []:
            try:
                report.fairness.append(self.fairness(feat))
            except Exception:
                pass

        if include_surrogate:
            try:
                report.surrogate = self.surrogate()
            except Exception:
                pass

        if include_quality and report.local_explanations:
            try:
                first = report.local_explanations[0]
                report.explanation_quality = evaluate_local(
                    self.model, self.X, first, self.problem_type
                )
            except Exception:
                pass

        report.summary = build_summary(report)
        return report


def explain_model(
    model: Any,
    X: Any,
    y: Optional[Any] = None,
    sensitive_features: Optional[Sequence[str]] = None,
    feature_names: Optional[list[str]] = None,
    problem_type: Optional[str] = None,
    explain_indices: Optional[Sequence[int]] = None,
    n_local: int = 3,
    top_k: Optional[int] = 10,
    use_shap: bool = True,
) -> ExplanationReport:
    """Produce a complete explainability report in a single call.

    Parameters
    ----------
    model : any fitted estimator with ``predict`` (and ideally ``predict_proba``).
    X : features (pandas DataFrame or array-like).
    y : optional ground-truth labels; enables metrics, permutation importance and
        the label-aware fairness diagnostics.
    sensitive_features : column names to run bias analysis on (e.g. ``["gender"]``).
    explain_indices : specific rows to explain locally; defaults to the first ``n_local``.
    """
    explainer = ModelExplainer(model, X, y, feature_names, problem_type, use_shap=use_shap)
    return explainer.report(
        sensitive_features=sensitive_features,
        explain_indices=explain_indices,
        n_local=n_local,
        top_k=top_k,
    )
