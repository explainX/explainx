"""Tests for the data-centric accuracy diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from explainx import ModelExplainer


@pytest.fixture
def clf_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, n_redundant=0, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), y


def test_error_analysis(clf_data):
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    ea = ModelExplainer(model, X, y).error_analysis()
    assert ea.metric == "misclassification_rate"
    assert 0.0 <= ea.baseline_error <= 1.0
    for s in ea.slices:
        assert s.lift >= 1.5 and s.size >= 10
    assert ea.recommendation
    json.dumps(ea.to_dict())


def test_label_issues_detects_injected_noise(clf_data):
    X, y = clf_data
    y_noisy = y.copy()
    rng = np.random.RandomState(1)
    flip = rng.choice(len(y), size=40, replace=False)
    y_noisy[flip] = 1 - y_noisy[flip]  # inject 10% label noise

    model = LogisticRegression(max_iter=500)
    li = ModelExplainer(model.fit(X, y_noisy), X, y_noisy).label_issues()
    assert li.out_of_sample is True  # cross-val path used for sklearn clf
    assert li.n_issues > 0
    # Flagged rows should overlap the truly-flipped ones well above chance.
    flagged = {it.index for it in li.issues}
    overlap = len(flagged & set(flip.tolist()))
    chance = len(flagged) * len(flip) / len(y)  # expected overlap if random
    assert overlap > 1.5 * chance and overlap >= 5
    json.dumps(li.to_dict())


def test_leakage_detection():
    rng = np.random.RandomState(0)
    n = 400
    noise = rng.normal(size=(n, 3))
    y = (noise[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    X = pd.DataFrame(noise, columns=["a", "b", "c"])
    X["leak"] = y  # a copy of the label = textbook leakage
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)

    rep = ModelExplainer(model, X, y).leakage()
    assert "leak" in rep.suspected_leakage
    assert rep.recommendation


def test_calibration_report(clf_data):
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    cal = ModelExplainer(model, X, y).calibration()
    assert 0.0 <= cal.ece <= 1.0
    assert cal.bins
    assert isinstance(cal.well_calibrated, bool)
    assert cal.recommendation
    json.dumps(cal.to_dict())


def test_diagnostics_work_through_adapter():
    """Diagnostics should work on a non-sklearn model via the adapter."""
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0, random_state=0)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    booster = xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(Xdf, label=y), num_boost_round=25)
    ex = ModelExplainer(__import__("explainx").wrap_model(booster, task="classification"), Xdf, y)
    ea = ex.error_analysis()
    cal = ex.calibration()
    assert ea.baseline_error >= 0
    assert cal.bins
