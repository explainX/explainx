from imports import *
from plotly_graphs import *
from protodash import *
"""
This class calculates feature importance

Input: 


"""


class dashboard():
    def __init__(self):
        super(dashboard, self).__init__()
        self.param= None

    def find(self,  df, y_variable,y_variable_predict, mode):
        self.available_columns= available_columns = list(df.columns)
        original_variables = [col for col in df.columns if '_impact' in col]
        self.impact_variables = [col for col in original_variables if not '_rescaled' in col]
        self.y_variable= y_variable
        self.y_variable_predict= y_variable_predict

        d= self.dash(df, mode)

        return True

    def dash(self, df, mode):
        y_variable= self.y_variable


        g = plotly_graphs()
        y_variables = [col for col in df.columns if '_impact' in col]
        original_variables = [col for col in df.columns if not '_impact' in col]
        original_variables = [col for col in original_variables if not '_rescaled' in col]
        array = []

        row_number = 1
        # y_variable = "predict"
        columns = ['index', '0', '1', '2', '3', '4']
        dat = []

        available_columns = list(df.columns)

        external_stylesheets = ['https://codepen.io/rab657/pen/LYpKraq.css',
                                {
                                    'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css',
                                    'rel': 'stylesheet'
                                }
                                ]

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

        app.title = "explainX.ai - Main Dashboard"

        navbar = dbc.Navbar(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(dbc.NavbarBrand("explainX.ai DASHBOARD", className="ml-2"))
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="https://plot.ly",
                ),

                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(dbc.NavbarBrand("Explain any machine learning model", className="ml-4"))
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="https://plot.ly",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),

            ],
            color="dark",
            dark=True,
        )

        app.layout = html.Div([
            navbar,
            html.Div([
                dbc.Button(
                    "View Data",
                    id="collapse-button",
                    className="mb-3",
                    color="primary"),

                dbc.Collapse(html.Div([
                    html.H4('Data',
                            style={'backgroundColor': '#fff',
                                   'color': 'black',
                                   'border-radius': '15px 15px 0px 0px ',
                                   'textAlign': 'left',
                                   'height': "50px",
                                   'margin': 'auto',
                                   'padding-left': '70px',
                                   "font-family": "Helvetica, Arial, sans-serif",

                                   }),
                    html.Div([
                        dash_table.DataTable(
                            id='datatable-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
                            ],
                            data=df.to_dict('records'),
                            editable=True,
                            filter_action="native",
                            sort_mode="multi",
                            row_selectable="multi",
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            style_table={'overflowX': 'auto',
                                         'margin': 'auto',
                                         "padding-left": '10px',
                                         'width': '90%'},
                            style_header={
                                'backgroundColor': '#0984e3',
                                'fontWeight': 'normal',
                                "fontSize": "15px",
                                'marginLeft': "10px",
                                'color': 'white'
                            },
                            style_cell={
                                "font-family": "Helvetica, Arial, sans-serif",
                                "fontSize": "11px",
                                'width': '{}%'.format(len(df.columns)),
                                'textOverflow': 'ellipsis',
                                'overflow': 'hidden',
                                'textAlign': 'left',
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                            },
                            css=[
                                {
                                    'selector': '.dash-spreadsheet td div',

                                    'rule': '''
                                    line-height: 15px;
                                    max-height: 20px; min-height: 20px; height: 20px;
                                    display: block;
                                    overflow-y: hidden;

                                '''
                                }])
                    ])

                ], style={'marginTop': 0}), id="collapse")],
                style={'backgroundColor': "#fff", 'marginTop': 20, 'marginBottom': 20}),

            # end of collapsable div
            dcc.Tabs([
                dcc.Tab(label='Global Explanation', children=[

                    # Global Feature Impact & Importance
                    html.Div(id='datatable-interactivity-container', children=[
                        html.Div([
                            html.Div([
                                html.H4('Global Feature Importance',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 15px 15px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px'

                                               }),
                                html.P(
                                    'Feature importance assign a score to input features based on how useful they are at predicting a target variable. ',
                                    style={'backgroundColor': '#fff',
                                           'color': 'black',
                                           'border-radius': '15px 15px 0px 0px ',
                                           'textAlign': 'left',
                                           'height': "50px",
                                           'margin': 'auto',
                                           'padding-left': '70px',
                                           'padding-top': '20px',
                                           "font-family": "Helvetica, Arial, sans-serif",

                                           })
                                ,
                                dcc.Loading(
                                    id="loading-1",
                                    type="circle",
                                    children=dcc.Graph(
                                        id="global_feature_importance",
                                        style={'marginLeft': 50, 'marginTop': 0, 'height': '500px'}

                                    )
                                )
                                , ], style={'backgroundColor': "#fff", "minHeight": "400px",
                                            'border-radius': '15px 15px 15px 15px',
                                            'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                            'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50'},
                            ),

                            html.Div([
                                html.H4('Global Feature Impact',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 15px 15px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px'

                                               }),
                                html.P(
                                    'Feature impact identifies which features (also known as columns or inputs) in a dataset have the greatest positive or negative effect on the outcomes of a machine learning model.',
                                    style={'backgroundColor': '#fff',
                                           'color': 'black',
                                           'border-radius': '15px 15px 0px 0px ',
                                           'textAlign': 'left',
                                           'height': "50px",
                                           'margin': 'auto',
                                           'padding-left': '70px',
                                           'padding-top': '20px',
                                           "font-family": "Helvetica, Arial, sans-serif",

                                           }),
                                dcc.Loading(
                                    id="loading-2",
                                    type="circle",
                                    children=dcc.Graph(id='global_feature_impact',
                                                       style={'marginLeft': 50, 'marginTop': 0, 'height': '500px'})
                                )

                            ],
                                style={'backgroundColor': "#fff", "minHeight": "400px",
                                       'border-radius': '15px 15px 15px 15px',
                                       'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                       'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                       'marginTop': 50},
                            ),
                        ],

                            style={
                                "marginTop": 50,
                                'border-radius': '15px 15px 15px 15px',

                            }
                        )
                    ], style={'height': '400'})
                ]),

                dcc.Tab(label='Feature Interaction', children=[

                    dcc.Loading(
                        id="feature_interaction_load",
                        type="circle",
                        children=html.Div([
                            html.Div([
                                html.H4('Partial Dependence Plot',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 15px 15px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px'

                                               }),
                                html.P(
                                    'The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model',
                                    style={'backgroundColor': '#fff',
                                           'color': 'black',
                                           'border-radius': '15px 15px 0px 0px ',
                                           'textAlign': 'left',
                                           'height': "50px",
                                           'margin': 'auto',
                                           'padding-left': '70px',
                                           'padding-top': '20px',
                                           "font-family": "Helvetica, Arial, sans-serif",

                                           }),
                                html.Div([
                                    html.P('Variable Name'),
                                    dcc.Dropdown(
                                        id='xaxis-column',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[1],
                                        clearable=False
                                    ),

                                ],
                                    style={'width': '25%', 'marginLeft': 70, 'float': 'left',
                                           'display': 'inline-block'}),

                                html.Div([
                                    html.P("Variable Impact Values"),
                                    dcc.Dropdown(
                                        id='yaxis-column',
                                        options=[{'label': i, 'value': i} for i in y_variables],
                                        value=y_variables[1],
                                        clearable=False
                                    ),

                                ], style={'width': '25%', 'float': 'center', 'display': 'inline-block'}),

                                html.Div([
                                    html.P('3rd Variable'),
                                    dcc.Dropdown(
                                        id='third-axis',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[-3],
                                        clearable=False
                                    ),

                                ], style={'width': '25%', 'marginRight': 10, 'float': 'center',
                                          'display': 'inline-block'}),

                            ]),
                            dcc.Loading(
                                id="loading-5",
                                type="circle",
                                children=dcc.Graph(id='indicator-graphic', style={'marginLeft': 50})
                            ),
                        ],
                            style={'backgroundColor': "#fff", "minHeight": "600px",
                                   'border-radius': '15px 15px 15px 15px',
                                   'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                   'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                   'marginTop': 50, 'width': '95%'}),
                    ),

                    dcc.Loading(
                        id="loading-2-pdp",
                        type='circle',
                        children=html.Div([
                            html.Div([
                                html.H4('Summary Plot',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 15px 15px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px'

                                               }),
                                html.P(
                                    'In the summary plot, we see first indications of the relationship between the value of a feature and the impact on the prediction',
                                    style={'backgroundColor': '#fff',
                                           'color': 'black',
                                           'border-radius': '15px 15px 0px 0px ',
                                           'textAlign': 'left',
                                           'height': "50px",
                                           'margin': 'auto',
                                           'padding-left': '70px',
                                           'padding-top': '20px',
                                           "font-family": "Helvetica, Arial, sans-serif",

                                           })
                                ,
                                dcc.Loading(
                                    id="loading-3",
                                    type="circle",
                                    children=dcc.Graph(id='summary_plot', style={'marginLeft': 50, 'height': '600px'})

                                ),

                            ], style={'backgroundColor': "#fff", "minHeight": "600px",
                                      'border-radius': '15px 15px 15px 15px',
                                      'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                      'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                      'marginTop': 50, }),
                        ], style={'marginBottom': 50,
                                  'marginTop': 50,
                                  'marginLeft': "1%",
                                  'width': '95%', })
                    )

                ]),

                # tab 3:Distribution

                dcc.Tab(label='Distribution', children=[

                    html.Div([

                        html.Div([
                            html.Div([
                                html.H4('Distributions',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 15px 15px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px'

                                               }),
                                html.Div([
                                    html.P('Variable Name'),
                                    dcc.Dropdown(
                                        id='xaxis-column-name',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[0]
                                    ),
                                ],
                                    style={'width': '25%', 'marginLeft': 70, 'float': 'left',
                                           'display': 'inline-block'}),
                                html.Div([
                                    html.P('Plot Type'),
                                    dcc.Dropdown(
                                        id='plot_type',
                                        options=[{'label': 'Violin Plot', 'value': 'Violin Plot'},
                                                 {'label': 'Histogram', 'value': 'Histogram'}, ],
                                        value='Histogram'
                                    ),
                                ],
                                    style={'width': '25%', 'marginLeft': 70, 'float': 'left',
                                           'display': 'inline-block'}),

                            ]),

                            dcc.Graph(id='indicator-graphic2',
                                      style={'marginLeft': 50, 'marginTop': 100, 'height': '500px'}),
                        ],
                            style={'backgroundColor': "#fff", "minHeight": "700px",
                                   'border-radius': '15px 15px 15px 15px',
                                   'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                   'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                   'marginTop': 50}),

                    ], style={'marginBottom': 50,
                              'marginTop': 50,
                              'marginLeft': "1%",
                              'width': '95%', })

                ]),
                # tab 4:Local Explanation

                dcc.Tab(label='Local Explanation', children=[

                    html.Div([

                        # Input and Data
                        html.Div([

                            html.Div([
                                html.H4('Data',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 0px 0px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px',
                                               "font-family": "Helvetica, Arial, sans-serif",

                                               }),
                                html.I("Index of Row That You Want To Explain.", style={'padding-left': '70px'}),

                                dcc.Input(id="row_number", type="number", value=1,
                                          placeholder="Enter a Row Number e.g. 1, 4, 5",
                                          style={'text-align': 'center'}),
                                html.Div([
                                    dash_table.DataTable(
                                        id='data_table_row',
                                        columns=[
                                            {"name": i, "id": i, "deletable": False, "selectable": True} for i in
                                            df.columns
                                        ],
                                        data=[],
                                        editable=True,
                                        sort_mode="multi",
                                        row_deletable=False,
                                        page_action="native",
                                        page_current=0,
                                        page_size=1,
                                        style_table={'overflowX': 'auto', 'margin': 'auto', "padding-left": '50px',
                                                     'width': '90%'},
                                        style_header={
                                            'backgroundColor': '#0984e3',
                                            'fontWeight': 'normal',
                                            "fontSize": "15px",
                                            'marginLeft': "10px",
                                            'color': 'white'
                                        },
                                        style_cell={
                                            "font-family": "Helvetica, Arial, sans-serif",
                                            "fontSize": "11px",
                                            'width': '{}%'.format(len(df.columns)),
                                            'textOverflow': 'ellipsis',
                                            'overflow': 'hidden',
                                            'textAlign': 'left',
                                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                        },
                                        css=[
                                            {
                                                'selector': '.dash-spreadsheet td div',

                                                'rule': '''
                            line-height: 15px;
                            max-height: 20px; min-height: 20px; height: 20px;
                            display: block;
                            overflow-y: hidden;

                        '''
                                            }],

                                    )
                                ], style={'marginLeft': '-30px'})

                            ], style={'backgroundColor': "#fff", "minHeight": "200px",
                                      'border-radius': '15px 15px 15px 15px',
                                      'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                      'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                      'marginTop': 50}),

                        ]),  # End of Input & Data Div

                        dcc.Tabs([
                            dcc.Tab(label='Local Feature Explanation', children=[

                                html.Div(id='datatable-interactivity-container2', children=[
                                    html.Div([
                                        html.Div([
                                            html.H4('Local Feature Importance',
                                                    style={'backgroundColor': '#fff',
                                                           'color': 'black',
                                                           'border-radius': '15px 15px 15px 15px ',
                                                           'textAlign': 'left',
                                                           'height': "50px",
                                                           'margin': 'auto',
                                                           'padding-left': '70px',
                                                           'padding-top': '20px'

                                                           }),
                                            html.P(
                                                'Feature importance assign a score to input features based on how useful they are at predicting a target variable. ',
                                                style={'backgroundColor': '#fff',
                                                       'color': 'black',
                                                       'border-radius': '15px 15px 0px 0px ',
                                                       'textAlign': 'left',
                                                       'height': "50px",
                                                       'margin': 'auto',
                                                       'padding-left': '70px',
                                                       'padding-top': '20px',
                                                       "font-family": "Helvetica, Arial, sans-serif",

                                                       })
                                            ,
                                            dcc.Loading(
                                                id="local_feature_importance_1",
                                                type="circle",
                                                children=dcc.Graph(
                                                    id="local_feature_importance",
                                                    style={'marginLeft': 50, 'marginTop': 0, 'height': '500px'})
                                            )
                                            , ], style={'backgroundColor': "#fff", "minHeight": "400px",
                                                        'border-radius': '15px 15px 15px 15px',
                                                        'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                                        'border-right': '1px solid #2c3e50',
                                                        'border-bottom': '1px solid #2c3e50'},
                                        ),

                                        html.Div([
                                            html.H4('Local Feature Impact',
                                                    style={'backgroundColor': '#fff',
                                                           'color': 'black',
                                                           'border-radius': '15px 15px 15px 15px ',
                                                           'textAlign': 'left',
                                                           'height': "50px",
                                                           'margin': 'auto',
                                                           'padding-left': '70px',
                                                           'padding-top': '20px'

                                                           }),
                                            html.P(
                                                'Local Feature impact identifies which features have the greatest positive or negative effect on the outcome of a machine learning model for a specific row.',
                                                style={'backgroundColor': '#fff',
                                                       'color': 'black',
                                                       'border-radius': '15px 15px 0px 0px ',
                                                       'textAlign': 'left',
                                                       'height': "50px",
                                                       'margin': 'auto',
                                                       'padding-left': '70px',
                                                       'padding-top': '20px',
                                                       "font-family": "Helvetica, Arial, sans-serif",

                                                       }),
                                            dcc.Loading(
                                                id="local_feature_impact_1",
                                                type="circle",
                                                children=dcc.Graph(id='local_feature_impact',
                                                                   style={'marginLeft': 50, 'marginTop': 0,
                                                                          'height': '500px'})
                                            )

                                        ],
                                            style={'backgroundColor': "#fff", "minHeight": "400px",
                                                   'border-radius': '15px 15px 15px 15px',
                                                   'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                                   'border-right': '1px solid #2c3e50',
                                                   'border-bottom': '1px solid #2c3e50', 'marginTop': 50},
                                        ),
                                    ],

                                        style={
                                            "marginTop": 50,
                                            'border-radius': '15px 15px 15px 15px'})], style={'height': '400'})

                            ]),

                            dcc.Tab(id="similar_tabs", label='Similar Profiles', children=[

                                html.Div([
                                    html.H4('Similar Profiles',
                                            style={'backgroundColor': '#fff',
                                                   'color': 'black',
                                                   'border-radius': '15px 15px 0px 0px ',
                                                   'textAlign': 'left',
                                                   'height': "50px",
                                                   'margin': 'auto',
                                                   'padding-left': '70px',
                                                   'padding-top': '20px',
                                                   "font-family": "Helvetica, Arial, sans-serif",

                                                   }),

                                    html.Button('Display Similar Profiles', id='btn-nclicks-1', n_clicks=0,
                                                style={'backgroundColor': '#fff',
                                                       'color': 'black',
                                                       'border-radius': '15px 15px 15px 15px ',
                                                       'textAlign': 'left',
                                                       'height': "50px",
                                                       'margin': 'auto',
                                                       'margin-left': '70px',
                                                       "font-family": "Helvetica, Arial, sans-serif",

                                                       }),

                                    html.P(
                                        'Display similar  profiles and the extent to which they are similar to the chosen applicant as indicated by the last row in the table below labelled as "Weight"',
                                        style={'backgroundColor': '#fff',
                                               'color': 'black',
                                               'border-radius': '15px 15px 0px 0px ',
                                               'textAlign': 'left',
                                               'height': "50px",
                                               'margin': 'auto',
                                               'padding-left': '70px',
                                               'padding-top': '20px',
                                               "font-family": "Helvetica, Arial, sans-serif",

                                               }),
                                    html.Div([
                                        dcc.Loading(
                                            id="table",
                                            type="circle",
                                            children=dash_table.DataTable(
                                                id='prototype_data',
                                                columns=[{"name": i, "id": i} for i in columns],
                                                data=[],
                                                editable=False,
                                                # filter_action="native",
                                                sort_mode="multi",
                                                row_deletable=False,
                                                page_action="native",
                                                page_current=0,
                                                page_size=len(original_variables) + 1,
                                                style_table={'overflowX': 'scroll', 'margin': 'auto',
                                                             "padding-left": '30px', 'width': '95%'},
                                                style_header={
                                                    'backgroundColor': '#0984e3',
                                                    'fontWeight': 'bold',
                                                    "fontSize": "15px",
                                                    'marginLeft': "10px",
                                                    'textAlign': 'center',
                                                    'color': 'white'
                                                },
                                                style_cell={
                                                    "font-family": "Helvetica, Arial, sans-serif",
                                                    "fontSize": "15px",
                                                    'width': '{}%'.format(len(df.columns)),
                                                    'textOverflow': 'ellipsis',
                                                    'overflow': 'hidden',
                                                    'textAlign': 'left',
                                                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                                    "marginBottom": "100px"
                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {

                                                            'filter_query': '{index} = "Weight(%)"'

                                                        },
                                                        "backgroundColor": "green",
                                                        'color': 'white'

                                                    }

                                                ],
                                                #
                                            )
                                        )

                                    ])

                                ], style={'backgroundColor': "#fff", "minHeight": "800px",
                                          'border-radius': '15px 15px 15px 15px',
                                          'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                          'border-right': '1px solid #2c3e50', 'border-bottom': '1px solid #2c3e50',
                                          'marginTop': 50, 'width': '100%'}),

                            ])

                        ], style={'marginTop': 50}), ])

                ])

            ]), ], style={'marginBottom': 50,
                          'marginTop': 10,
                          'marginLeft': "1%",
                          'marginRight': "1%",
                          'width': '97%'})

        @app.callback(
            Output('datatable-interactivity', 'style_data_conditional'),
            [Input('datatable-interactivity', 'selected_columns')]
        )
        def update_styles(selected_columns):
            return [{
                'if': {'column_id': i},
                'background_color': '#D2F3FF'
            } for i in selected_columns]

        @app.callback(
            Output("collapse", "is_open"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        # Global Feature Importance
        @app.callback(
            Output('global_feature_importance', "figure"),
            [Input('datatable-interactivity', "derived_virtual_data"),
             Input('datatable-interactivity', "derived_virtual_selected_rows")])
        def update_graphs(rows, derived_virtual_selected_rows):
            dff = df if rows is None else pd.DataFrame(rows)
            g = plotly_graphs()
            fig, __ = g.feature_importance(dff)
            time.sleep(1)
            return fig

        @app.callback(
            Output('global_feature_impact', "figure"),
            [Input('datatable-interactivity', "derived_virtual_data"),
             Input('datatable-interactivity', "derived_virtual_selected_rows")])
        def update_graphs(rows, derived_virtual_selected_rows):
            dff = df if rows is None else pd.DataFrame(rows)
            g = plotly_graphs()
            fig, __ = g.feature_impact(dff)
            time.sleep(1)
            return fig

        # Local Eeature Importance

        @app.callback(
            Output(component_id='data_table_row', component_property='data'),
            [Input(component_id='row_number', component_property='value')])
        def update_table(row_number):
            i = 0
            if type(row_number) == type(1):
                i = row_number
            array = df[i:i + 1]
            array = array.to_dict('records')
            return array

        # Partial Dependence Plot Graph
        @app.callback(
            Output('indicator-graphic', 'figure'),
            [Input('xaxis-column', 'value'),
             Input('yaxis-column', 'value'),
             Input('third-axis', 'value'),
             Input('datatable-interactivity', "derived_virtual_data"),
             Input('datatable-interactivity', "derived_virtual_selected_rows")
             ])
        def update_graph(xaxis_column_name, yaxis_column_name, third_axis_name, rows, derived_virtual_selected_rows):
            df3 = df if rows is None else pd.DataFrame(rows)
            fig = {
                'data': [dict(
                    x=df3[xaxis_column_name],
                    y=df3[yaxis_column_name],
                    text=df3[third_axis_name],
                    mode='markers',
                    marker={
                        'size': 15,
                        'opacity': 1,
                        'line': {'width': 2, 'color': 'DarkSlateGrey'},
                        'color': df3[third_axis_name],
                    }
                )],
                'layout': dict(
                    xaxis={
                        'title': xaxis_column_name,
                        'margin': {'l': 40, 'b': 20, 't': 300, 'r': 0},
                    },
                    yaxis={
                        'title': yaxis_column_name,

                    },
                    margin={'l': 40, 'b': 140, 't': 50, 'r': 140},
                    hovermode='closest',
                    legend={'x': 0, 'y': 1},
                    color_continuous_scale=px.colors.cyclical.IceFire,
                    color_discrete_sequence=["red", "blue"]
                )
            }
            time.sleep(1)
            return fig

        # Local Feature Importance & Impact
        @app.callback(
            Output('local_feature_importance', "figure"),
            [Input('data_table_row', "data")])
        def update_impact_graph(data):
            data = pd.DataFrame(data)
            fig, __ = g.feature_importance(data)
            time.sleep(1)
            return fig

        @app.callback(
            Output('local_feature_impact', "figure"),
            [Input('data_table_row', "data")])
        def update_impact_graph(data):
            data = pd.DataFrame(data)
            fig, __ = g.feature_impact(data)
            time.sleep(1)
            return fig

        # Prototypical Analysis
        @app.callback(
            Output(component_id='prototype_data', component_property='data'),
            [Input(component_id='row_number', component_property='value'),
             Input('btn-nclicks-1', 'n_clicks')])
        def update_table(row_number, btn1):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'btn-nclicks-1' in changed_id:
                p = protodash()
                p.preprocess_data(df, y_variable)
                dfs = p.find_prototypes(row_number)
                dat = dfs.T.reset_index()
                dat = dat.to_dict('records')
                return dat
            else:
                return []

        # Summary Plot
        @app.callback(
            Output('summary_plot', 'figure'),
            [Input('datatable-interactivity', "derived_virtual_data")
             ])
        def update_graph2(rows):
            dff = df if rows is None else pd.DataFrame(rows)
            sp, __ = g.summary_plot(dff)
            time.sleep(1)
            return sp

        # Distributions
        @app.callback(
            Output('indicator-graphic2', 'figure'),
            [Input('xaxis-column-name', 'value'),
             Input('plot_type', 'value')
             ])
        def update_graph2(xaxis_column_name, plot_type):
            df3 = df
            # For categorical variables only
            cat_variables = []
            num_variables = []
            for i in original_variables:
                if df[i].dtype == 'object':
                    cat_variables.append(i)
                else:
                    num_variables.append(i)

            if plot_type == "Histogram":
                group_labels = ['xxaxis_column_name']
                return px.histogram(df3, x=xaxis_column_name, marginal="box")
            else:
                for i in cat_variables:
                    return px.violin(df3, x=xaxis_column_name, box=True, points='all')
                else:
                    return px.violin(df3, y=xaxis_column_name, box=True, points='all')

        if mode=="inline":
            app.run_server(mode="inline")
        else:
            app.run_server()

        return True






