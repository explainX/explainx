from .imports import *
from .rescale_numeric_feature import *

import catboost
from catboost import CatBoostClassifier,Pool
from catboost import CatBoostRegressor
from catboost.utils import get_confusion_matrix

"""
This class calculates feature importance

Input: 


"""


class calculate_shap():
    def __init__(self):
        super(calculate_shap, self).__init__()
        self.param = None

    def xgboost_shap(self, model, X):
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        pd_shap = pd.DataFrame(shap_values)
        all_columns = list(X.columns)

        shap_columns = []

        for i in all_columns:
            shap_columns.append(i + "_impact")

        pd_shap.columns = shap_columns

        Y = X.join(pd_shap)

        return Y

    def catboost_shap(self, model, df, y_variable=None):
        # explain the model's predictions using SHAP
        if y_variable != None:
            try:
                df = df.drop(y_variable, axis=1)
            except:
                df = df

        # find categorical variables and it's index
        g = get_cols()

        cat, cat_index = g.cate_col_with_index(df)
        all_columns = list(df.columns)

        # convert dataframe in to array
        df_array = df.to_numpy()

        # call the function
        shap_values = self.get_shap_values(df_array, model, all_columns, cat_index)

        # append the results with the original file

        final_df = df.join(shap_values)

        return final_df

    def get_shap_values(self, x_array, model, x_variable, cat_index):
        """
        SHAP VALUES CALCULATED
        """
        shap_values = model.get_feature_importance(Pool(x_array, cat_features=cat_index), type='ShapValues')
        shap_values = shap_values[:, :-1]
        total_columns = x_variable
        total_columns = [i + '_impact' for i in total_columns]

        shap_values = pd.DataFrame(data=shap_values, columns=total_columns)
        return shap_values

    def find(self, model, df, model_name="xgboost"):

        if model_name == "xgboost":
            df2 = self.xgboost_shap(model, df)
            return df2

        elif model_name == "lightgbm":
            df2 = self.xgboost_shap(model, df)

            return df2

        elif model_name == "catboost":
            df2 = self.catboost_shap(model, df)
            return df2


