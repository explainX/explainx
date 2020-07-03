from imports import *

"""
This class calculates feature impact

Input: 


"""


class feature_impact():
    def __init__(self):
        super(feature_impact, self).__init__()
        self.param= None


    def find(self,  df):

        variables = [col for col in df.columns if '_impact' in col]
        y = []
        for i in range(len(variables)):
            p = df[variables[i]].mean()
            y.append(p)
        res = {variables[i]: y[i] for i in range(len(y))}
        res2 = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
        res3 = pd.Series(res2, name='Impact_Value')
        res3.index.name = 'VariableName'
        fi = res3.reset_index()
        return fi


