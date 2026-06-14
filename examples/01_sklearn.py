"""explainX with scikit-learn — works directly, no wrapping.

    python examples/01_sklearn.py
"""

from sklearn.ensemble import RandomForestClassifier

from explainx import explain_model, ModelExplainer
from _common import loan_data, show

X, y = loan_data()
model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X, y)

# Any scikit-learn estimator is consumed directly.
report = explain_model(model, X, y, sensitive_features=["gender"], n_local=1)
show(report, ModelExplainer(model, X, y))
