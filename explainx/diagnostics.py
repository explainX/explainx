"""Data-centric diagnostics that help *improve* model accuracy.

Explanations tell you how a model behaves; these diagnostics tell you what to
*fix* to make it more accurate -- the data-centric-AI playbook:

* **error_analysis** -- discover the data slices where the model fails worst, so
  you know where to add data, add features, or split the model.
* **find_label_issues** -- flag likely-mislabeled rows (confident learning);
  cleaning them is one of the highest-ROI accuracy levers (Northcutt et al.).
* **detect_leakage** -- find features that alone predict the target almost
  perfectly, a red flag for target leakage that inflates offline accuracy and
  collapses in production.
* **assess_calibration** -- measure whether predicted probabilities are
  trustworthy, and recommend a fix when they aren't.

All return structured, JSON-serializable reports with a plain-language
``recommendation`` an LLM agent can act on.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import (
    ErrorAnalysis, ErrorSlice, LabelIssues, LabelIssue,
    LeakageReport, LeakageFeature, CalibrationReport, CalibrationBin,
)
from .utils import predict, predict_scores


# --------------------------------------------------------------------------- #
# Error analysis / slice discovery
# --------------------------------------------------------------------------- #
def error_analysis(
    model: Any, X: pd.DataFrame, y: Any, problem_type: str,
    n_bins: int = 4, min_size: int = 10, min_lift: float = 1.5, top_k: int = 8,
) -> ErrorAnalysis:
    y = np.asarray(y)
    y_pred = predict(model, X)
    if problem_type == "classification":
        err = (y_pred != y).astype(float)
        metric = "misclassification_rate"
    else:
        err = np.abs(y.astype(float) - y_pred.astype(float))
        metric = "mean_absolute_error"
    baseline = float(err.mean())

    slices: list[ErrorSlice] = []
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
            edges = np.unique(np.quantile(series, np.linspace(0, 1, n_bins + 1)))
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                mask = (series >= lo) & (series <= hi) if i == len(edges) - 2 else (series >= lo) & (series < hi)
                cond = f"{lo:.3g} <= {col} <= {hi:.3g}" if i == len(edges) - 2 else f"{lo:.3g} <= {col} < {hi:.3g}"
                _add_slice(slices, mask.to_numpy(), err, col, cond, baseline, min_size, min_lift)
        else:
            for val in series.unique():
                mask = (series == val).to_numpy()
                _add_slice(slices, mask, err, col, f"{col} == {val}", baseline, min_size, min_lift)

    slices.sort(key=lambda s: s.lift * s.size, reverse=True)
    slices = slices[:top_k]
    rec = (
        f"Worst slice: '{slices[0].condition}' has {slices[0].lift:.1f}x the average error. "
        "Prioritise more/better data or features there, or train a specialised model for that segment."
        if slices else "No slice exceeded the error-lift threshold; error is spread fairly evenly."
    )
    return ErrorAnalysis(problem_type=problem_type, metric=metric, baseline_error=baseline,
                         slices=slices, recommendation=rec)


def _add_slice(slices, mask, err, col, cond, baseline, min_size, min_lift):
    n = int(mask.sum())
    if n < min_size:
        return
    slice_err = float(err[mask].mean())
    lift = slice_err / baseline if baseline > 0 else 0.0
    if lift >= min_lift:
        slices.append(ErrorSlice(feature=col, condition=cond, size=n,
                                 error=slice_err, baseline_error=baseline, lift=float(lift)))


# --------------------------------------------------------------------------- #
# Label-error detection (confident learning)
# --------------------------------------------------------------------------- #
def find_label_issues(
    model: Any, X: pd.DataFrame, y: Any, cv: int = 3, top_k: Optional[int] = 50,
) -> LabelIssues:
    y = np.asarray(y)
    classes = np.unique(y)
    class_index = {c: i for i, c in enumerate(classes)}

    proba, out_of_sample = _oos_proba(model, X, y, cv)
    if proba is None:
        raise ValueError("Label-issue detection needs a classifier with predict_proba.")

    # Per-class self-confidence thresholds (confident learning).
    self_conf = np.array([proba[i, class_index[y[i]]] for i in range(len(y))])
    thresholds = {c: float(self_conf[y == c].mean()) if np.any(y == c) else 0.0 for c in classes}

    issues: list[LabelIssue] = []
    for i in range(len(y)):
        argmax = int(np.argmax(proba[i]))
        suggested = classes[argmax]
        if suggested != y[i] and self_conf[i] < thresholds[y[i]]:
            issues.append(LabelIssue(index=int(i), given_label=_native(y[i]),
                                     suggested_label=_native(suggested), confidence=float(proba[i, argmax])))
    issues.sort(key=lambda it: it.confidence, reverse=True)
    n_issues = len(issues)
    if top_k:
        issues = issues[:top_k]
    rec = (
        f"~{n_issues} of {len(y)} labels look wrong ({n_issues/len(y):.1%}). Review the highest-confidence "
        "ones; relabelling or removing them typically improves accuracy more than model tuning."
        if n_issues else "No systematic label issues detected."
    )
    return LabelIssues(n_checked=int(len(y)), n_issues=n_issues,
                       estimated_noise_rate=float(n_issues / len(y)), out_of_sample=out_of_sample,
                       issues=issues, recommendation=rec)


def _oos_proba(model, X, y, cv):
    """Out-of-sample probabilities via cross-val when possible (less optimistic)."""
    try:
        from sklearn.base import clone, is_classifier
        from sklearn.model_selection import cross_val_predict

        if is_classifier(model):
            proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
            return np.asarray(proba), True
    except Exception:
        pass
    # Fallback: in-sample probabilities from the fitted model (optimistic).
    p = predict_scores_proba(model, X)
    return (None, False) if p is None else (p, False)


def predict_scores_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))
        except Exception:
            try:
                return np.asarray(model.predict_proba(X.to_numpy()))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Target-leakage detection
# --------------------------------------------------------------------------- #
def detect_leakage(
    model: Any, X: pd.DataFrame, y: Any, problem_type: str, threshold: Optional[float] = None,
) -> LeakageReport:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score

    y = np.asarray(y)
    is_clf = problem_type == "classification"
    metric = "roc_auc" if (is_clf and len(np.unique(y)) == 2) else ("accuracy" if is_clf else "r2")
    thr = threshold if threshold is not None else (0.95 if metric != "roc_auc" else 0.98)

    feats: list[LeakageFeature] = []
    for col in X.columns:
        xcol = pd.get_dummies(X[[col]], drop_first=False)
        try:
            est = DecisionTreeClassifier(max_depth=3, random_state=0) if is_clf else DecisionTreeRegressor(max_depth=3, random_state=0)
            score = float(np.mean(cross_val_score(est, xcol, y, cv=3, scoring=metric)))
        except Exception:
            score = 0.0
        feats.append(LeakageFeature(feature=col, solo_score=score, suspected=score >= thr))

    feats.sort(key=lambda f: f.solo_score, reverse=True)
    suspected = [f.feature for f in feats if f.suspected]
    rec = (
        f"Possible target leakage: {', '.join(suspected)} alone predict the target with "
        f"{metric} >= {thr}. Confirm these are available at prediction time and not a proxy for the label."
        if suspected else "No single feature predicts the target suspiciously well; no obvious leakage."
    )
    return LeakageReport(metric=metric, threshold=float(thr), features=feats,
                         suspected_leakage=suspected, recommendation=rec)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def assess_calibration(model: Any, X: pd.DataFrame, y: Any, n_bins: int = 10) -> CalibrationReport:
    proba = predict_scores_proba(model, X)
    if proba is None:
        raise ValueError("Calibration needs a classifier with predict_proba.")
    y = np.asarray(y)
    classes = getattr(model, "classes_", np.unique(y))

    conf = proba.max(axis=1)
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    correct = (pred == y).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        acc = float(correct[m].mean()); mc = float(conf[m].mean()); cnt = int(m.sum())
        bins.append(CalibrationBin(mean_confidence=mc, accuracy=acc, count=cnt))
        ece += (cnt / len(y)) * abs(acc - mc)

    # Brier score (binary uses positive-class prob; multiclass: one-vs-rest mean).
    if proba.shape[1] == 2:
        pos = classes[-1]
        brier = float(np.mean((proba[:, -1] - (y == pos).astype(float)) ** 2))
    else:
        onehot = np.zeros_like(proba)
        idx = {c: i for i, c in enumerate(classes)}
        for i, yi in enumerate(y):
            if yi in idx:
                onehot[i, idx[yi]] = 1.0
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    well = ece < 0.1
    rec = (
        "Probabilities are reasonably calibrated."
        if well else
        f"ECE={ece:.3f} indicates miscalibration; wrap the model in sklearn's "
        "CalibratedClassifierCV (isotonic or sigmoid/Platt) before trusting probabilities."
    )
    return CalibrationReport(ece=float(ece), brier=brier, well_calibrated=well,
                             bins=bins, recommendation=rec)


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
