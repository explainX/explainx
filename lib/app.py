import dash
from imports import *
from plotly_graphs import *
from plotly_css import *

external_stylesheets = ['https://raw.githubusercontent.com/rab657/explainx/master/explainx.css', dbc.themes.BOOTSTRAP,

                          
                            {
                                'href':'https://fonts.googleapis.com/css?family=Montserrat',
                                'rel' :'stylesheet'
                            }
                            ]

# external JavaScript files
external_scripts = ['https://raw.githubusercontent.com/explainX/explainx/share_button/hotjar.js',
                   {'src': 'https://raw.githubusercontent.com/explainX/explainx/share_button/hotjar.js'}]

app = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True, external_scripts=external_scripts)

app.title = "explainX.ai - Main Dashboard"


server = app.server
