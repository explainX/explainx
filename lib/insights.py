from imports import *
from insight_classification import *
from insight_regression import *

"""
This class calculates feature importance

Input: 


"""


class insights():
    def __init__(self,param):
        super(insights, self).__init__()
        self.param = param
        self.regression = insight_regression()
        self.classification = insight_classification()

    def insight_1_feature_imp(self, df):
        if self.param["is_classification"] == True:
            return self.classification.insight_1_feature_imp(df)
        else:
            return self.regression.insight_1_feature_imp(df)

    def insight_2_global_feature_impact(self, df, outcome=0):
        if self.param["is_classification"] == True:
            return self.classification.insight_2_global_feature_impact(df, outcome, self.param["expected_values"],
                                                                       self.param["classes"])
        else:
            return self.regression.insight_2_global_feature_impact(df, self.param["expected_values"][0])
    
    def insight_2_local_feature_impact(self, df, outcome=0):
        if self.param["is_classification"] == True:
            return self.classification.insight_2_local_feature_impact(df, outcome, self.param["expected_values"],
                                                                       self.param["classes"])
        else:
            return self.regression.insight_2_global_feature_impact(df, self.param["expected_values"][0])

    def insight_3(self, df):
        if self.param["is_classification"] == True:
            return self.classification.insight_3(df)
        else:
            return self.regression.insight_3(df)



