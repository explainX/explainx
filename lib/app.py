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
        <script>
            (function(h,o,t,j,a,r){
                h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
                h._hjSettings={hjid:2146099,hjsv:6};
                a=o.getElementsByTagName('head')[0];
                r=o.createElement('script');r.async=1;
                r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
                a.appendChild(r);
            })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
        </script>
    </head>
    <body>
       
        {%app_entry%}
        <footer>
            {%config%}
           
            {%renderer%}
        </footer>
       
    </body>
</html>
'''

app.title = "explainX.ai - Main Dashboard"


server = app.server
