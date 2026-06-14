"""explainX with statsmodels — wrap the fitted results with wrap_model.

statsmodels' .predict returns the positive-class probability (Logit) or the
predicted value (OLS), with no sklearn API, so wrap_model adapts it. Pass
task= so explainX knows whether it's classification or regression.

    pip install statsmodels
    python examples/07_statsmodels.py
"""

import numpy as np
import statsmodels.api as sm

from explainx import ModelExplainer, wrap_model
from _common import loan_data

X, y = loan_data()
Xc = sm.add_constant(X)  # statsmodels needs an explicit intercept column

logit = sm.Logit(y, Xc).fit(disp=0)

# The wrapped predict() must receive the same columns the model was fit on, so
# we re-add the constant inside a custom predict_proba_fn.
def _design(Z):
    import pandas as pd
    Z = Z if hasattr(Z, "columns") else pd.DataFrame(Z, columns=X.columns)
    return sm.add_constant(Z, has_constant="add")

ex = ModelExplainer(
    wrap_model(predict_proba_fn=lambda Z: logit.predict(_design(Z)), classes=[0, 1]),
    X, y,
)
print(ex.metrics().to_dict()["metrics"])
print("Top features:", [f.feature for f in ex.importance().top(3)])
print(ex.fairness("gender").findings[0])
