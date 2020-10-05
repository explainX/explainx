from explainx import *

from catboost import CatBoostRegressor
import pandas as pd
from xgboost import XGBClassifier as xgb
import xgboost
from sklearn.model_selection import train_test_split

# train XGBoost model
X,y = explainx.dataset_heloc()


#xgboost
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

explainx.ai_test(X, y, model, model_name="xgboost")

from sklearn.ensemble import GradientBoostingRegressor

# Load boston dataset
X,y = explainx.dataset_boston()

# split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# start and stop explainx
explainx.ai_test(X_test,  y_test, model, model_name="gradientboostingregressor")

#test other functions that find all the graphs.


# Load Heloc dataset
X,y = explainx.dataset_heloc()

# Split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# Run catboost model
model = CatBoostRegressor(iterations=150,
                          learning_rate=.3,
                          depth=2)


# Fit model
model.fit(X_train.to_numpy(), y_train)

explainx.ai_test(X_test,  y_test, model, model_name="catboost")
