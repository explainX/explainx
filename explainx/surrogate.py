"""Global surrogate: a glassbox decision tree that mimics the black box.

We fit a shallow decision tree to reproduce the model's predictions, then read
off its rules. Unlike importance scores, the surrogate gives an *exact,
inspectable* approximation of the decision logic -- and a **fidelity** score
saying how faithfully it reproduces the original model, so you know how much to
trust the rules.

This is one of the four most-deployed XAI techniques (alongside SHAP, LIME and
counterfactuals) and the spiritual cousin of InterpretML's glassbox models.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .schema import SurrogateExplanation, FeatureImportance
from .utils import predict


def global_surrogate(
    model: Any,
    X: pd.DataFrame,
    problem_type: str,
    max_depth: int = 4,
) -> SurrogateExplanation:
    y_model = predict(model, X)
    X_enc = pd.get_dummies(X, drop_first=False)

    if problem_type == "classification":
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.metrics import accuracy_score

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree.fit(X_enc, y_model)
        fidelity = float(accuracy_score(y_model, tree.predict(X_enc)))
        fidelity_metric = "accuracy"
    else:
        from sklearn.tree import DecisionTreeRegressor, export_text
        from sklearn.metrics import r2_score

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(X_enc, y_model.astype(float))
        fidelity = float(r2_score(y_model.astype(float), tree.predict(X_enc)))
        fidelity_metric = "r2"

    rules_text = export_text(tree, feature_names=list(X_enc.columns), max_depth=max_depth)

    importances = np.asarray(tree.feature_importances_, dtype=float)
    feats = [
        FeatureImportance(feature=col, importance=float(importances[i]))
        for i, col in enumerate(X_enc.columns)
    ]
    feats.sort(key=lambda f: abs(f.importance), reverse=True)

    return SurrogateExplanation(
        method="decision_tree",
        fidelity=fidelity,
        fidelity_metric=fidelity_metric,
        max_depth=max_depth,
        rules_text=rules_text,
        feature_importances=feats,
    )
