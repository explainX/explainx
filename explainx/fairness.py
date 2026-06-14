"""Bias / fairness analysis across a sensitive attribute.

Given a sensitive feature (e.g. ``gender``), this compares model outcomes
across its groups and computes the standard group-fairness diagnostics:

* **selection rate** per group -- P(model predicts the positive outcome).
* **demographic parity difference** -- max gap in selection rate between groups
  (0 is perfectly fair).
* **disparate impact ratio** -- min/max selection rate. The US "four-fifths
  rule" flags ratios below 0.8 as evidence of adverse impact.
* **equal-opportunity difference** -- max gap in true-positive rate (requires
  ground-truth labels), i.e. does the model find genuine positives equally well
  across groups.

This directly answers the question in the vision: *"is my model rejecting one
group more than another regardless of profile?"*
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import FairnessReport, GroupFairness
from .utils import predict, positive_class

FOUR_FIFTHS = 0.8
PARITY_WARN = 0.1


def fairness_analysis(
    model: Any,
    X: pd.DataFrame,
    sensitive_feature: str,
    y_true: Optional[Any] = None,
    problem_type: str = "classification",
    pos_label: Any = None,
) -> FairnessReport:
    if sensitive_feature not in X.columns:
        raise ValueError(f"sensitive_feature '{sensitive_feature}' not in data columns")

    y_pred = predict(model, X)
    groups_values = X[sensitive_feature]
    y_true_arr = None if y_true is None else np.asarray(y_true)

    report = FairnessReport(
        sensitive_feature=sensitive_feature,
        problem_type=problem_type,
    )

    if problem_type == "regression":
        return _regression_fairness(report, groups_values, y_pred, y_true_arr)

    pos = pos_label if pos_label is not None else positive_class(model, y_true)
    return _classification_fairness(report, groups_values, y_pred, y_true_arr, pos)


def _classification_fairness(report, groups_values, y_pred, y_true, pos) -> FairnessReport:
    selection_rates: dict[Any, float] = {}
    tpr_by_group: dict[Any, float] = {}

    for group in pd.unique(groups_values):
        mask = (groups_values == group).to_numpy()
        n = int(mask.sum())
        group_pred = y_pred[mask]
        sel_rate = float(np.mean(group_pred == pos))
        selection_rates[group] = sel_rate

        gf = GroupFairness(group=_native(group), count=n, selection_rate=sel_rate)

        if y_true is not None:
            group_true = y_true[mask]
            gf.accuracy = float(np.mean(group_pred == group_true))
            actual_pos = group_true == pos
            actual_neg = ~actual_pos
            if actual_pos.sum() > 0:
                tpr = float(np.mean(group_pred[actual_pos] == pos))
                gf.true_positive_rate = tpr
                tpr_by_group[group] = tpr
            if actual_neg.sum() > 0:
                gf.false_positive_rate = float(np.mean(group_pred[actual_neg] == pos))

        report.groups.append(gf)

    rates = list(selection_rates.values())
    if rates:
        report.demographic_parity_difference = float(max(rates) - min(rates))
        nonzero_max = max(rates)
        report.disparate_impact_ratio = float(min(rates) / nonzero_max) if nonzero_max > 0 else None
    if len(tpr_by_group) >= 2:
        tprs = list(tpr_by_group.values())
        report.equal_opportunity_difference = float(max(tprs) - min(tprs))

    _flag_classification(report, selection_rates, pos)
    return report


def _flag_classification(report, selection_rates, pos):
    findings: list[str] = []
    di = report.disparate_impact_ratio
    dp = report.demographic_parity_difference

    if di is not None and di < FOUR_FIFTHS:
        report.biased = True
        lo = min(selection_rates, key=selection_rates.get)
        hi = max(selection_rates, key=selection_rates.get)
        findings.append(
            f"Disparate impact ratio {di:.2f} is below the 0.8 four-fifths threshold: "
            f"group '{lo}' receives the positive outcome ({pos}) at "
            f"{selection_rates[lo]:.1%} vs '{hi}' at {selection_rates[hi]:.1%}."
        )
    if dp is not None and dp >= PARITY_WARN:
        if not findings:
            report.biased = True
        findings.append(
            f"Demographic parity gap of {dp:.1%} between groups exceeds the "
            f"{PARITY_WARN:.0%} warning threshold."
        )
    if report.equal_opportunity_difference is not None and report.equal_opportunity_difference >= PARITY_WARN:
        report.biased = True
        findings.append(
            f"True-positive-rate gap of {report.equal_opportunity_difference:.1%} across groups: "
            "the model finds genuine positives less reliably for some groups (equal-opportunity violation)."
        )
    if not findings:
        findings.append("No group-fairness threshold was breached on the metrics evaluated.")
    report.findings = findings


def _regression_fairness(report, groups_values, y_pred, y_true) -> FairnessReport:
    means: dict[Any, float] = {}
    for group in pd.unique(groups_values):
        mask = (groups_values == group).to_numpy()
        gp = y_pred[mask].astype(float)
        mean_pred = float(np.mean(gp)) if len(gp) else None
        means[group] = mean_pred
        gf = GroupFairness(group=_native(group), count=int(mask.sum()), mean_prediction=mean_pred)
        report.groups.append(gf)

    vals = [v for v in means.values() if v is not None]
    if vals:
        gap = float(max(vals) - min(vals))
        spread = float(np.mean(np.abs(y_pred.astype(float))) + 1e-9)
        report.demographic_parity_difference = gap
        if gap > 0.1 * spread:
            report.biased = True
            lo = min(means, key=means.get)
            hi = max(means, key=means.get)
            report.findings = [
                f"Average prediction differs across groups: '{lo}'={means[lo]:.3g} vs "
                f"'{hi}'={means[hi]:.3g} (gap {gap:.3g})."
            ]
        else:
            report.findings = ["Average predictions are comparable across groups."]
    return report


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
