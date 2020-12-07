import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash

# from app import app
from .apps import local_explanation


class localExplanation():
    def __init__(self, x_test, ShapleyValues=None, df_with_shap=None):
        self.data = x_test
        self.df_with_shap = df_with_shap
        self.shapley_values = ShapleyValues

    def main_function(self, mode):

        external_stylesheets = ['https://raw.githubusercontent.com/rab657/explainx/master/explainx.css',
                                dbc.themes.BOOTSTRAP,
                                {
                                    'href': 'https://fonts.googleapis.com/css?family=Montserrat',
                                    'rel': 'stylesheet'
                                }
                                ]
        local = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

        local.title = "explainX.ai - Local Level Explanation"

        local.layout = local_explanation.layout_local(self.shapley_values, self.data, self.df_with_shap, local)
        debug_value = False
        if mode is None:
            import random
            port = random.randint(6000, 7000)
            return local.run_server(port=port, debug=debug_value, dev_tools_ui=debug_value,
                                    dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                    dev_tools_hot_reload=True)
        else:
            import random
            port = random.randint(6000, 7000)
            return local.run_server(mode='inline', port=port, debug=debug_value, dev_tools_ui=debug_value,
                                    dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                    dev_tools_hot_reload=True)
