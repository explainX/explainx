from imports import *
from feature_impact import *
from feature_importance import *
from shap_pdp import *
from summary_plot import *

class data_for_shap_graphs():
    def __init__(self):
        super(data_for_shap_graphs, self).__init__()
        self.param= None

        # save all important variables here.


    def feature_importance(self,  df):
        fimp = feature_importance()
        df3 = fimp.find(df)
        return df3

    def feature_impact(self, df):
        fi = feature_impact()
        df2 = fi.find(df)
        return df2

    def summary_plot(self, df):
        sp = summary_plot()
        df5 = sp.find(df)
        return df5

    def partial_dependence_plot(self, df):
        pdp = shap_pdp()
        df4 = pdp.find(df)
        return df4
