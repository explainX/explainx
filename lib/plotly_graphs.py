from imports import *
from feature_impact import *
from feature_importance import *
from shap_pdp import *
from summary_plot import *
from data_for_shap_graphs import *


class plotly_graphs():
    def __init__(self):
        super(plotly_graphs, self).__init__()
        self.data = data_for_shap_graphs()

        # save all important variables here.

    def feature_importance_old(self, df):
        df2 = self.data.feature_importance(df)

        names = list(df2["VariableName"])
        new_names = []

        for n in names:
            new_names.append(n.strip("_impact"))

        df2["VariableName"] = new_names

        feature_importance = px.bar(df2, x='Impact_Value', y="VariableName", orientation='h')
        return feature_importance, df2

    def feature_importance(self, df, classification=False):
        if classification==True:
            df2 = self.data.feature_importance_classification(df)

            names = list(df2["VariableName"])
            new_names = []

            for n in names:
                new_names.append(n.strip("_impact"))

            df2["VariableName"] = new_names
            return df2
        else:

            df2 = self.data.feature_importance(df)

            names = list(df2["VariableName"])
            new_names = []

            for n in names:
                new_names.append(n.strip("_impact"))

            df2["VariableName"] = new_names

            # feature_importance = px.bar(df2, x='Impact_Value', y="VariableName", orientation='h')
            # return feature_importance, df2
            return df2

    def feature_importance_graph(self, df):
        feature_importance = px.bar(df, x='VariableName', y="Impact_Value", orientation='v')
        return feature_importance

    def feature_impact_old(self, df):
        df2 = self.data.feature_impact(df)

        names = list(df2["VariableName"])
        new_names = []

        for n in names:
            new_names.append(n.strip("_impact"))

        df2["VariableName"] = new_names

        # Feature Impact Plot
        feature_impact = px.bar(df2, x='Impact_Value', y="VariableName", orientation='h')
        return feature_impact, df2

    def feature_impact(self, df, classification=False):

        if classification == True:
            df2 = self.data.feature_impact_classification(df)

            names = list(df2["VariableName"])
            new_names = []

            for n in names:
                new_names.append(n.strip("_impact"))

            df2["VariableName"] = new_names

            return df2

        else:


            df2 = self.data.feature_impact(df)

            names = list(df2["VariableName"])
            new_names = []

            for n in names:
                new_names.append(n.strip("_impact"))

            df2["VariableName"] = new_names
            return df2

    def feature_impact_graph(self, df):
        feature_impact = px.bar(df, x='Impact_Value', y="VariableName", orientation='h')
        return feature_impact

    def summary_plot(self, df,classification=False):
        df2 = self.data.summary_plot(df)

        summary_plot = px.scatter(df2, x="Feature Impact on Outcome", y="Feature Name", color="Rescaled Feature Value",
                                  hover_data=["Original Feature Value"], color_continuous_scale="Bluered_r", template="plotly_white")

        # return summary_plot, df2
        return df2

    def summary_plot_graph(self, df,classification=False):
        summary_plot = px.scatter(df, x="Feature Impact on Outcome", y="Feature Name", color="Rescaled Feature Value",
                                  hover_data=["Original Feature Value"], color_continuous_scale="Bluered_r", template="plotly_white")
        return summary_plot

    def partial_dependence_plot(self, df, v1=None, v2=None, v3=None):
        pdp = shap_pdp()
        df = pdp.find(df)
        # add if and else so we can colors to discrete variables
        # g = px.scatter(df, x=v1, y=v2, color=v3, color_continuous_scale="RdBu",
        #                color_discrete_sequence=px.colors.sequential.Plasma_r)
        # # return g, df
        return df

    def pdp_plot(self, df, v1, v2, v3):
        g = px.scatter(df, x=v1, y=v2, color=v3, color_continuous_scale="Bluered_r",
                       color_discrete_sequence=px.colors.sequential.Plasma_r, template="plotly_white")
        return g

    def distributions(self, df, variable_name):
        graph = px.histogram(df, x=variable_name, marginal="box", template="plotly_white")
        return graph

    def global_feature_impact_graph(self, df,classification=False):

        if classification==True:

            fig = px.bar(df, x="VariableName", y="Impact_Value",
                         color='class_name', barmode='group',
                         height=400, template= 'plotly_white')

            return fig
        else:
            signs_ = ['+' if i > 0 else '-' for i in df['Impact_Value']]
            colors_ = ['rgba(41, 128, 185,1.0)' if i > 0 else 'rgba(192, 57, 43,1.0)' for i in df['Impact_Value']]
            feature_impact = go.Figure()
            feature_impact.add_trace(go.Bar(
                y=list(df['Impact_Value']),
                x=list(df['VariableName']),
                name='features_impact',
                text=signs_,
                texttemplate="%{y:.2f} ",
                textposition='outside',
                cliponaxis=False,
                orientation='v',
                hovertemplate='Feature: %{y} <br> Impact: %{y:.4f}<extra></extra>',
                marker=dict(
                    color=colors_,
                    line=dict(color='rgba(0, 0, 0, 0.5)', width=0))))
            #feature_impact.update_layout({'xaxis': {'title': 'Impact on Prediction'}})
            feature_impact.update_layout({'yaxis': {'title': 'Impact on Model Output'}})

        return feature_impact
    
    def global_feature_importance_graph(self, df, classification=False):

        if classification==True:

            fig = px.bar(df, x="VariableName", y="Impact_Value",
                         color='class_name', barmode='group',
                         height=400, template= 'plotly_white')

            return fig

        else:
            feature_impact = go.Figure()
            feature_impact.add_trace(go.Bar(
                y=list(df['Impact_Value']),
                x=list(df['VariableName']),
                name='features_impact',
                #text=signs_,
                texttemplate="%{y:.2f} ",
                textposition='auto',
                cliponaxis=False,
                orientation='v',
                hovertemplate='Feature: %{y} <br> Impact: %{y:.4f}<extra></extra>',
                marker=dict(
                    #color=colors_,
                    line=dict(color='rgba(0, 0, 0, 0.5)', width=0))))
            #feature_impact.update_layout({'xaxis': {'title': 'Impact on Prediction'}})
            feature_impact.update_layout({'yaxis': {'title': 'Average Impact on Model Output'}})

            return feature_impact


    def local_feature_impact_graph(self, df):
        df2 = self.data.feature_impact(df)
        names = list(df2["VariableName"])
        new_names = []
        for n in names:
            new_names.append(n.strip("_impact"))
        df2["VariableName"] = new_names
        df2.sort_values(by='Impact_Value', ascending=True, inplace=True)
        signs_ = ['+' if i > 0 else '-' for i in df2['Impact_Value']]
        colors_ = ['rgba(41, 128, 185,1.0)' if i > 0 else 'rgba(192, 57, 43,1.0)' for i in df2['Impact_Value']]
        feature_impact = go.Figure()
        feature_impact.add_trace(go.Bar(
            y=list(df2['VariableName']),
            x=list(df2['Impact_Value']),
            name='features_impact',
            text=signs_,
            texttemplate="%{x:.2f} ",
            textposition='auto',
            cliponaxis=False,
            orientation='h',
            hovertemplate='Feature: %{y} <br> Impact: %{x:.4f}<extra></extra>',
            marker=dict(
                color=colors_,
                line=dict(color='rgba(0, 0, 0, 0.5)', width=0))))
        feature_impact.update_layout({'xaxis': {'title': 'Impact on Prediction'}})
        feature_impact.update_layout({'yaxis': {'title': 'Feature Name'}})
        feature_impact.update_layout(template="plotly_white")

        return feature_impact, df2
