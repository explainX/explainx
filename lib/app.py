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

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        {%scripts%}
    </head>
    <body>
        <div>My Custom header</div>
        {%app_entry%}
        <footer>
            {%config%}
           
            {%renderer%}
        </footer>
        <div>My Custom footer</div>
    </body>
</html>
'''

app.title = "explainX.ai - Main Dashboard"


server = app.server
