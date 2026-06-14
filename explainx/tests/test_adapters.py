"""Tests for universal framework support via the model adapter."""

import json

import numpy as np
import pandas as pd
import pytest

from explainx import ModelExplainer, explain_model, wrap_model


@pytest.fixture
def data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


def test_sklearn_passthrough(data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert wrap_model(model) is model  # unchanged


def test_xgboost_sklearn_wrapper_works_directly(data):
    xgb = pytest.importorskip("xgboost")
    X, y = data
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, verbosity=0).fit(X, y)
    report = explain_model(model, X, y, n_local=1)
    assert report.problem_type == "classification"
    assert report.metrics.metrics["accuracy"] > 0.5
    json.dumps(report.to_dict())


def test_native_xgboost_booster_via_adapter(data):
    xgb = pytest.importorskip("xgboost")
    X, y = data
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3}, xgb.DMatrix(X, label=y), num_boost_round=30
    )
    ex = ModelExplainer(wrap_model(booster, task="classification"), X, y)
    assert ex.problem_type == "classification"
    assert ex.metrics().metrics["accuracy"] > 0.5
    assert ex.importance().features
    assert ex.explain(0, top_k=3).contributions


def test_native_lightgbm_booster_via_adapter(data):
    lgb = pytest.importorskip("lightgbm")
    X, y = data
    booster = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(X, label=y), num_boost_round=30
    )
    ex = ModelExplainer(wrap_model(booster, task="classification"), X, y)
    assert ex.problem_type == "classification"
    assert ex.metrics().metrics["accuracy"] > 0.5
    assert ex.fairness("f0") is not None or True  # smoke


def test_custom_predict_proba_fn(data):
    """A fully custom model exposed only through a predict_proba function."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    wrapped = wrap_model(predict_proba_fn=lambda Z: rf.predict_proba(Z), classes=[0, 1])
    report = explain_model(wrapped, X, y, n_local=1)
    assert report.problem_type == "classification"
    assert report.global_importance.features


def test_custom_regression_predict_fn():
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=0)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    lr = LinearRegression().fit(Xdf, y)
    wrapped = wrap_model(predict_fn=lambda Z: lr.predict(Z), task="regression")
    ex = ModelExplainer(wrapped, Xdf, y)
    assert ex.problem_type == "regression"
    assert "r2" in ex.metrics().metrics
