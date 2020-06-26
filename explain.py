import os
import sys
from pathlib import Path


path= Path(__file__).parent.absolute()
path_dataset= os.path.join(path, "datasets")
path= os.path.join(path, "lib")

sys.path.append(path)


from imports import *
from dashboard import *
from calculate_shap import *
"""
This class calculates feature importance

Input: 


"""


class explain():
    def __init__(self):
        super(explain, self).__init__()
        self.param= None

    # is classification function?

    def is_classification_given_y_array(self, y_test):
        is_classification = False
        total = len(y_test)
        total_unique = len(set(y_test))
        if total < 30:
            if total_unique < 10:
                is_classification = True
        else:
            if total_unique < 20:
                is_classification = True
        return is_classification


    def ai(self,  df,  y, model, model_name="xgboost", mode=None):
        y_variable= "y_actual"
        y_variable_predict= "y_prediction"


        # is classification?
        is_classification= self.is_classification_given_y_array(y)

        # If yes, then different shap functuions are required.
        # get the shap value based on predcton and make a new dataframe.

        # find predictions first as shap values need that.

        prediction_col=[]

        if model_name == "xgboost":
            prediction_col = model.predict(xgboost.DMatrix(df))

        elif model_name == "catboost":
            prediction_col = model.predict(df.to_numpy())

        else:
            prediction_col = model.predict(df.to_numpy())


        #shap
        c = calculate_shap()
        self.df_final = c.find(model, df, prediction_col, is_classification, model_name=model_name)

        #prediction col
        self.df_final[y_variable_predict] = prediction_col



        self.df_final[y_variable] = y

        d= dashboard()
        d.find(self.df_final, y_variable, y_variable_predict, mode)

        return True

    def dataset_boston(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.boston()
        return X,y


    def dataset_iris(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.iris()
        return X,y


    def dataset_heloc(self):
        dataset= pd.read_csv(path_dataset+"/heloc_dataset.csv")

        map_riskperformance= {"RiskPerformance": {"Good":1, "Bad":0}}
        dataset.replace(map_riskperformance, inplace=True)
        y= list(dataset["RiskPerformance"])
        X= dataset.drop("RiskPerformance", axis=1)
        return X,y





explainx=explain()




