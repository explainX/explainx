from imports import *

"""
This class calculates feature importance

Input: 


"""


class feature_importance_classification():
    def __init__(self):
        super(feature_importance_classification, self).__init__()
        self.param= None


    def find(self,  df):

        df_shap = pd.DataFrame()
        classes = list(df["y_prediction"].unique())

        for c in classes:
            df_f= df[df['y_prediction'] == c]
            df_shap0= self.average_shap(df_f)
            df_shap0["class_name"]= str(c)

            df_shap= df_shap.append(df_shap0)


        return df_shap

    def average_shap(self, df):

        variables = [col for col in df.columns if '_impact' in col]
        y = []
        for i in range(len(variables)):
            p = df[variables[i]].abs().mean()
            y.append(p)
        res = {variables[i]: y[i] for i in range(len(y))}
        res2 = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
        res3 = pd.Series(res2, name='Impact_Value')
        res3.index.name = 'VariableName'
        fi = res3.reset_index()
        return fi


