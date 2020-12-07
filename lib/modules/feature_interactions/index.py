from explainx.lib.modules.feature_interactions.apps.feature_interaction import layout_interaction
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash

#from app import app
from .apps import feature_interaction


class featureInteraction():
    def __init__(self, x_test, df_with_shap=None):
        self.data = x_test
        self.df_with_shap = df_with_shap
        
    def main_function(self, mode):
        
        external_stylesheets = ['https://raw.githubusercontent.com/rab657/explainx/master/explainx.css', 
                                dbc.themes.BOOTSTRAP,
                                    {
                                        'href':'https://fonts.googleapis.com/css?family=Montserrat',
                                        'rel' :'stylesheet'
                                    }
                                    ]
        
        local = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

        local.title = "explainX.ai - Feature Interaction"
        
        local.layout = feature_interaction.layout_interaction(self.data, self.df_with_shap, local)
        
        if mode == None:
            import random
            port = random.randint(5000,6000)
            return local.run_server(port=port)
        else:
            import random
            port = random.randint(5000,6000)
            return local.run_server(mode='inline', port=port)
       