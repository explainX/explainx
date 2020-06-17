from imports import *
from feature_impact import *
from feature_importance import *
from shap_pdp import *
from summary_plot import *
from data_for_shap_graphs import *

class plotly_graphs():
    def __init__(self):
        super(plotly_graphs, self).__init__()
        self.data= data_for_shap_graphs()

        # save all important variables here.


    def feature_importance(self,  df):
        df2 = self.data.feature_importance(df)

        names = list(df2["VariableName"])
        new_names = []

        for n in names:
            new_names.append(n.strip("_impact"))

        df2["VariableName"] = new_names

        feature_importance = px.bar(df2, x='Impact_Value', y="VariableName", orientation='h', title='Feature Importance',)
        return feature_importance, df2

    def feature_impact(self, df):
        df2 = self.data.feature_impact(df)

        names = list(df2["VariableName"])
        new_names = []

        for n in names:
            new_names.append(n.strip("_impact"))

        df2["VariableName"] = new_names

        # Feature Impact Plot
        feature_impact = px.bar(df2, x='Impact_Value', y="VariableName", orientation='h', title='Global Feature Impact',
                                )
        return feature_impact, df2

    def summary_plot(self, df):
        df2 = self.data.summary_plot(df)

        summary_plot = px.scatter(df2, x="xaxis", y="yaxis", color="color", hover_data=["hover"])

        return summary_plot, df2

    def partial_dependence_plot(self, df, v1, v2, v3):
        pdp = shap_pdp()
        df = pdp.find(df)
        g= px.scatter(df, x=v1, y=v2, color=v3)
        return g, df


    def distributions(self, df, variable_name):
        graph= px.histogram(df, x=variable_name, marginal="box")
        return graph



