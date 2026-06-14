"""explainX with LightGBM — sklearn API and native Booster.

    pip install lightgbm
    python examples/03_lightgbm.py
"""

import lightgbm as lgb

from explainx import explain_model, ModelExplainer, wrap_model
from _common import loan_data, show

X, y = loan_data()

# 1) sklearn API (LGBMClassifier) — works directly.
print("=== LGBMClassifier (sklearn API) ===")
clf = lgb.LGBMClassifier(n_estimators=60, verbose=-1).fit(X, y)
report = explain_model(clf, X, y, sensitive_features=["gender"], n_local=1)
show(report, ModelExplainer(clf, X, y))

# 2) native Booster — wrap it.
print("\n=== native lightgbm.Booster (via wrap_model) ===")
booster = lgb.train({"objective": "binary", "verbose": -1},
                    lgb.Dataset(X, label=y), num_boost_round=60)
ex = ModelExplainer(wrap_model(booster, task="classification"), X, y)
print(ex.metrics().to_dict()["metrics"])
