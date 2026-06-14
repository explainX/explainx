"""explainX with CatBoost — works directly (CatBoost follows the sklearn API).

    pip install catboost
    python examples/04_catboost.py
"""

from catboost import CatBoostClassifier

from explainx import explain_model, ModelExplainer
from _common import loan_data, show

X, y = loan_data()
model = CatBoostClassifier(iterations=120, depth=4, verbose=0).fit(X, y)

# CatBoostClassifier has predict / predict_proba, so explainX uses it directly.
report = explain_model(model, X, y, sensitive_features=["gender"], n_local=1)
show(report, ModelExplainer(model, X, y))
