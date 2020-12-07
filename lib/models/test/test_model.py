# Copyright (c) 2020 explainX.ai
# Distributed under the MIT software license

import pytest
from modelprocessor import ModelProcessor

from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

classifiers = [RandomForestClassifier(),  MLPClassifier()]
regressors = [RandomForestRegressor(),AdaBoostRegressor(), MLPRegressor(), GradientBoostingRegressor()]

def test_classification():
    try:
        for model in classifiers:
            assert ModelProcessor().is_classification(model) == True
    except Exception:
        pytest.fail("Not a classifier!")

def test_not_classification():
    try:
        for model in regressors:
            assert ModelProcessor().is_classification(model) == False
    except Exception:
        pytest.fail("Not a regressor!")




