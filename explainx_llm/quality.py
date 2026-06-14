"""Quantify how much to trust an explanation itself.

A 2024-2025 theme in XAI research is that explanations must be *evaluated*, not
assumed correct. We compute two of the most established metrics for a local
explanation:

* **Faithfulness** -- do the attributions actually reflect the model? We remove
  features (reset to baseline) and check that the predicted-score drop
  correlates with the attributed importance. High correlation => faithful.
* **Stability (robustness)** -- do tiny, label-preserving input perturbations
  yield similar attributions? Measured as mean cosine similarity between the
  original attribution vector and perturbed ones. High similarity => stable.

These let an agent decide whether an explanation is reliable enough to act on.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from .schema import ExplanationQuality, LocalExplanation
from .utils import predict_scores, baseline_row


def evaluate_local(
    model: Any,
    X: pd.DataFrame,
    explanation: LocalExplanation,
    problem_type: str,
    n_perturbations: int = 20,
    random_state: int = 0,
) -> ExplanationQuality:
    notes: list[str] = []
    faithfulness = _faithfulness(model, X, explanation, problem_type, notes)
    stability = _stability(
        model, X, explanation, problem_type, n_perturbations, random_state, notes
    )
    return ExplanationQuality(
        method=explanation.method,
        faithfulness=faithfulness,
        stability=stability,
        notes=notes,
    )


def _faithfulness(model, X, explanation, problem_type, notes) -> Optional[float]:
    baseline = baseline_row(X)
    instance = X.iloc[[explanation.index]]
    base_score = predict_scores(model, instance, problem_type)
    if base_score is None:
        notes.append("Faithfulness unavailable: model exposes no continuous score.")
        return None
    base_score = float(base_score[0])

    attributed, observed = [], []
    for c in explanation.contributions:
        perturbed = instance.copy()
        perturbed[c.feature] = baseline[c.feature]
        s = predict_scores(model, perturbed, problem_type)
        if s is None:
            continue
        attributed.append(abs(c.contribution))
        observed.append(abs(base_score - float(s[0])))

    if len(attributed) < 2 or np.std(attributed) == 0 or np.std(observed) == 0:
        notes.append("Faithfulness undefined: not enough variation in attributions.")
        return None
    corr = float(np.corrcoef(attributed, observed)[0, 1])
    return corr


def _stability(model, X, explanation, problem_type, n_perturbations, random_state, notes) -> Optional[float]:
    from .local import explain_instance

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if not numeric_cols:
        notes.append("Stability unavailable: no numeric features to perturb.")
        return None

    order = [c.feature for c in explanation.contributions]
    base_vec = _vector(explanation, order)
    if np.linalg.norm(base_vec) == 0:
        notes.append("Stability undefined: zero attribution vector.")
        return None

    rng = np.random.RandomState(random_state)
    stds = X[numeric_cols].std().replace(0, 1.0).to_numpy()
    row = explanation.index
    base_numeric = X.loc[row, numeric_cols].to_numpy(dtype=float)
    sims = []
    for _ in range(n_perturbations):
        perturbed = X.copy()
        # Cast to float so adding continuous noise to int-coded columns is safe.
        perturbed[numeric_cols] = perturbed[numeric_cols].astype(float)
        noise = rng.normal(0, 0.1, size=len(numeric_cols)) * stds
        perturbed.loc[row, numeric_cols] = base_numeric + noise
        try:
            new_exp = explain_instance(model, perturbed, row, problem_type, use_shap=False)
        except Exception:
            continue
        vec = _vector(new_exp, order)
        denom = np.linalg.norm(base_vec) * np.linalg.norm(vec)
        if denom > 0:
            sims.append(float(np.dot(base_vec, vec) / denom))

    if not sims:
        return None
    return float(np.mean(sims))


def _vector(explanation: LocalExplanation, order: list[str]) -> np.ndarray:
    lookup = {c.feature: c.contribution for c in explanation.contributions}
    return np.array([lookup.get(f, 0.0) for f in order], dtype=float)
