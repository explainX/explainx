from explainx.lib.imports import *
from .insight_classification import *
from .insight_regression import *
from explainx.lib.utils import is_classification

"""
This class calculates feature importance

Input: 


"""


class insights():
    def __init__(self,model):
        super(insights, self).__init__()

        self.model = model
        self.regression = insight_regression()
        self.classification = insight_classification()

    def insight_1_feature_imp(self, df):
        if is_classification(self.model) == True:
            return self.classification.insight_1_feature_imp(df)
        else:
            return self.regression.insight_1_feature_imp(df)

    def insight_2_global_feature_impact(self, df, outcome=0):
        if is_classification(self.model) == True:
            return self.classification.insight_2_global_feature_impact(df, outcome, self.param["expected_values"], self.param["classes"])
        else:
            return self.regression.insight_2_global_feature_impact(df, self.param["expected_values"][0])
    
    def insight_2_local_feature_impact(self, df, y_and_prob):
        if is_classification(self.model) == True:
            return self.classification.insight_2_local_feature_impact(df, y_and_prob)
        else:
            return self.regression.insight_2_local_feature_impact(df, y_and_prob)

    def insight_3(self, df):
        if is_classification(self.model) == True:
            return self.classification.insight_3(df)
        else:
            return self.regression.insight_3(df)



