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
            return df2

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


    def summary_plot(self, df,classification=False):
        df2 = self.data.summary_plot(df)

        summary_plot = px.scatter(df2, x="Feature Impact on Outcome", y="Feature Name", color="Rescaled Feature Value",
                                  hover_data=["Original Feature Value"], color_continuous_scale="Bluered_r", template="plotly_white")
        return df2


    def summary_plot_graph(self, df,classification=False):
        summary_plot = px.scatter(df, x="Feature Impact on Outcome", y="Feature Name", color="Rescaled Feature Value",
                                  hover_data=["Original Feature Value"], color_continuous_scale="Bluered_r", template="plotly_white")
        return summary_plot


    def partial_dependence_plot(self, df, v1=None, v2=None, v3=None):
        pdp = shap_pdp()
        df = pdp.find(df)
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
            feature_impact = go.Figure()
            for class_name, group in df.groupby("class_name"):
                feature_impact.add_trace(go.Bar(
                    x=group["VariableName"], 
                    y=group["Impact_Value"], 
                    name=class_name,
                    hovertemplate='Feature: %{x} <br> Impact: %{y:.4f}<extra></extra>'))
                feature_impact.update_layout(
                    margin={'t': 0},
                    template="plotly_white",
                    autosize=True,
                    yaxis=dict(
                        title_text="Impact on Model Output"),
                    font=dict(
                        size=10),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    legend_title_text='Class')
                feature_impact.update_xaxes(showgrid=False, automargin=True)
                feature_impact.update_yaxes(showgrid=True, automargin=True)
            return feature_impact
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
            feature_impact.update_layout({'yaxis': {'title': 'Impact on Model Output'}})
            return feature_impact
    

    def global_feature_importance_graph(self, df, classification=False):
        if classification==True:
            feature_importance = go.Figure()
            #for class_name, group in df.groupby("class_name"):
            feature_importance.add_trace(go.Bar(
                x=list(df["VariableName"]), 
                y=list(df["Impact_Value"]), 
                #name=df["class_name"],
                hovertemplate='Feature: %{x} <br> Impact: %{y:.4f}<extra></extra>'))
            
            feature_importance.update_layout(
                margin={'t': 0},
                template="plotly_white",
                autosize=True,
                yaxis=dict(
                    title_text="Impact on Model Output"),
                font=dict(
                    size=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                legend_title_text='Class')
            
            feature_importance.update_xaxes(showgrid=False, automargin=True)
            feature_importance.update_yaxes(showgrid=True, automargin=True)
            return feature_importance

        else:
            feature_impact = go.Figure()
            feature_impact.add_trace(go.Bar(
                y=list(df['Impact_Value']),
                x=list(df['VariableName']),
                name='features_impact',
                texttemplate="%{y:.2f} ",
                textposition='auto',
                cliponaxis=False,
                orientation='v',
                hovertemplate='Feature: %{y} <br> Impact: %{y:.4f}<extra></extra>',
                marker=dict(
                    line=dict(color='rgba(0, 0, 0, 0.5)', width=0))))
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
        feature_impact.update_layout(template="plotly_white", margin={'t': 0, }, font=dict(size=10),)

        return feature_impact, df2
