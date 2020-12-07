import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
from .apps import cohort_app


class cohort():
    def __init__(self, x_test, model):
        self.data = x_test
        self.model = model

    def main_function(self, mode):

        external_stylesheets = ['https://raw.githubusercontent.com/rab657/explainx/master/explainx.css',
                                dbc.themes.BOOTSTRAP,
                                {
                                    'href': 'https://fonts.googleapis.com/css?family=Montserrat',
                                    'rel': 'stylesheet'
                                }
                                ]
        cohort = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

        cohort.title = "explainX.ai - Model Performance Analysis"

        cohort.layout = cohort_app.test_func(self.data, self.model, cohort)
        debug_value = False
        if mode == None:
            import random
            port = random.randint(4000, 5000)
            return cohort.run_server(port=port, debug=debug_value, dev_tools_ui=debug_value,
                                     dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                     dev_tools_hot_reload=True)
        else:
            import random
            port = random.randint(4000, 5000)
            return cohort.run_server(mode='inline', port=port, debug=debug_value, dev_tools_ui=debug_value,
                                     dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                     dev_tools_hot_reload=True)
