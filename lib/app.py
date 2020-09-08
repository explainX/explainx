import dash
from imports import *
from plotly_graphs import *
from plotly_css import *

external_stylesheets = ['https://codepen.io/rab657/pen/LYpKraq.css',
                            {
                                'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css',
                                'rel': 'stylesheet'
                            }
                            ]

app = JupyterDash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.title = "explainX.ai - Main Dashboard"


server = app.server