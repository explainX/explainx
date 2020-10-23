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

app = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.title = "explainX.ai - Main Dashboard"


server = app.server