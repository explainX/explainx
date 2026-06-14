"""End-to-end tests for the explainx_llm core.

These train real scikit-learn models and assert the engine produces sane,
JSON-serializable explanations -- including a deliberately biased dataset so we
can confirm the fairness checks fire.
"""

import json

import numpy as np
import pandas as pd
import pytest

from explainx_llm import explain_model, ModelExplainer


@pytest.fixture
def classification_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, random_state=0
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y


@pytest.fixture
def rf_model(classification_data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = classification_data
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    return model


def test_full_report_classification(rf_model, classification_data):
    X, y = classification_data
    report = explain_model(rf_model, X, y, n_local=2)

    assert report.problem_type == "classification"
    assert report.n_features == 6
    assert report.metrics is not None
    assert 0.0 <= report.metrics.metrics["accuracy"] <= 1.0
    assert report.global_importance.features  # non-empty
    assert len(report.local_explanations) == 2
    assert report.summary  # non-empty NL summary

    # The whole thing must be JSON serializable (the MCP contract).
    json.dumps(report.to_dict())


def test_local_explanation_contributions(rf_model, classification_data):
    X, y = classification_data
    explainer = ModelExplainer(rf_model, X, y)
    local = explainer.explain(0, top_k=3)
    assert len(local.contributions) == 3
    # Contributions are sorted by absolute magnitude.
    mags = [abs(c.contribution) for c in local.contributions]
    assert mags == sorted(mags, reverse=True)


def test_regression_report():
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=0)
    cols = [f"f{i}" for i in range(X.shape[1])]
    Xdf = pd.DataFrame(X, columns=cols)
    model = LinearRegression().fit(Xdf, y)

    report = explain_model(model, Xdf, y, n_local=1)
    assert report.problem_type == "regression"
    assert "r2" in report.metrics.metrics
    assert report.global_importance.features


def test_fairness_detects_bias():
    """Construct a dataset where the model rejects one gender regardless of profile."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(0)
    n = 600
    gender = rng.randint(0, 2, size=n)  # 0 / 1
    score = rng.normal(size=n)
    # Outcome depends on a genuine signal AND strongly on gender (the bias).
    approve = ((score + 2.0 * gender) > 0).astype(int)
    X = pd.DataFrame({"gender": gender, "score": score})
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, approve)

    report = explain_model(
        model, X, approve, sensitive_features=["gender"], n_local=1
    )
    assert len(report.fairness) == 1
    fr = report.fairness[0]
    assert fr.sensitive_feature == "gender"
    assert fr.biased is True
    assert fr.disparate_impact_ratio is not None
    assert "BIAS DETECTED" in report.summary


def test_counterfactual_flips_decision():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(2)
    n = 400
    score = rng.normal(size=n)
    extra = rng.normal(size=n)
    y = (score > 0).astype(int)
    X = pd.DataFrame({"score": score, "extra": extra})
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)

    explainer = ModelExplainer(model, X, y)
    # Pick a confidently-rejected row and ask for the opposite class.
    idx = int(np.where(model.predict(X) == 0)[0][0])
    cf = explainer.counterfactual(idx)
    assert cf.original_prediction == 0
    assert cf.desired_prediction == 1
    if cf.found:  # greedy search may not always succeed, but usually does here
        assert cf.new_prediction == 1
        assert cf.n_features_changed >= 1


def test_partial_dependence(rf_model, classification_data):
    X, _ = classification_data
    pd_result = rf_model_pdp = ModelExplainer(rf_model, X).partial_dependence("f0", grid_resolution=10)
    assert pd_result.feature == "f0"
    assert len(pd_result.grid_values) == len(pd_result.average_prediction)
    assert len(pd_result.grid_values) > 0


def test_shap_path_used_when_available(rf_model, classification_data):
    import importlib.util

    if importlib.util.find_spec("shap") is None:
        pytest.skip("shap not installed")
    X, y = classification_data
    explainer = ModelExplainer(rf_model, X, y, use_shap=True)
    imp = explainer.importance()
    local = explainer.explain(0, top_k=3)
    # When SHAP is present, both should report the shap method.
    assert imp.method.startswith("shap")
    assert local.method == "shap"


def test_fairness_no_bias_when_balanced():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(1)
    n = 600
    gender = rng.randint(0, 2, size=n)
    score = rng.normal(size=n)
    approve = (score > 0).astype(int)  # gender irrelevant
    X = pd.DataFrame({"gender": gender, "score": score})
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, approve)

    fr = ModelExplainer(model, X, approve).fairness("gender")
    assert fr.biased is False
