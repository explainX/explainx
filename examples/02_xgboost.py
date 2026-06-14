"""explainX with XGBoost — both the sklearn API and the native Booster.

    pip install xgboost
    python examples/02_xgboost.py
"""

import xgboost as xgb

from explainx import explain_model, ModelExplainer, wrap_model
from _common import loan_data, show

X, y = loan_data()

# 1) sklearn API (XGBClassifier) — works directly, no wrapping.
print("=== XGBClassifier (sklearn API) ===")
clf = xgb.XGBClassifier(n_estimators=60, max_depth=3, verbosity=0).fit(X, y)
report = explain_model(clf, X, y, sensitive_features=["gender"], n_local=1)
show(report, ModelExplainer(clf, X, y))

# 2) native Booster (xgboost.train) — wrap it; tell explainX it's classification.
print("\n=== native xgboost.Booster (via wrap_model) ===")
booster = xgb.train({"objective": "binary:logistic", "max_depth": 3},
                    xgb.DMatrix(X, label=y), num_boost_round=60)
ex = ModelExplainer(wrap_model(booster, task="classification"), X, y)
print(ex.metrics().to_dict()["metrics"])
print("Top features:", [f.feature for f in ex.importance().top(3)])
