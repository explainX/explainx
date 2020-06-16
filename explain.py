from .imports import *
from .dashboard import *
from .calculate_shap import *
"""
This class calculates feature importance

Input: 


"""


class explain():
    def __init__(self):
        super(explain, self).__init__()
        self.param= None


    def ai(self,  df,  y, model, model_name="xgboost", mode=None):
        y_variable= "y_variable"
        y_variable_predict= "prediction"



        #shap
        c = calculate_shap()
        self.df_final = c.find(model, df, model_name=model_name)

        #prediction col
        if model_name=="xgboost":
            self.df_final[y_variable_predict] = model.predict(xgboost.DMatrix(df))

        if model_name=="catboost":
            self.df_final[y_variable_predict] = model.predict(df.to_numpy())

        self.df_final[y_variable] = y

        d= dashboard()
        d.find(self.df_final, y_variable, y_variable_predict, mode)

        return True

    def dataset_boston(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.boston()
        return X,y





explainx=explain()




