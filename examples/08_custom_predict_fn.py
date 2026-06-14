"""explainX with ANY model via a custom prediction function.

If your model isn't one of the known frameworks — a remote API, an ONNX
runtime, an ensemble, a hand-written rule — just give wrap_model a function.

* classification: pass predict_proba_fn (returns per-class probabilities)
* regression:     pass predict_fn (returns the predicted value)

    python examples/08_custom_predict_fn.py
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from explainx import ModelExplainer, wrap_model
from _common import loan_data

X, y = loan_data()

# Pretend this is an opaque model we can only call through a function.
_backend = GradientBoostingClassifier(random_state=0).fit(X, y)
def my_predict_proba(rows):
    """rows: a DataFrame or array -> (n, n_classes) probability matrix."""
    return _backend.predict_proba(rows)

ex = ModelExplainer(wrap_model(predict_proba_fn=my_predict_proba, classes=[0, 1]), X, y)
print(ex.metrics().to_dict()["metrics"])
print("Top features:", [f.feature for f in ex.importance().top(3)])
local = ex.explain(0, top_k=3)
print(f"Row 0 -> {local.prediction}: "
      + ", ".join(f"{c.feature} ({c.contribution:+.3f})" for c in local.contributions))
print(ex.fairness("gender").findings[0])
