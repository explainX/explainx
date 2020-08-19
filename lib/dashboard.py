from imports import *
from plotly_graphs import *
from protodash import *
from insights import *
from plotly_css import *
import pandasql as psql
import string
import random
import os


class dashboard():
    def __init__(self):
        super(dashboard, self).__init__()
        self.param = None
        self.query_dict = dict()
        self.filtered_dataframe = dict()
        self.feature_importance = dict()
        self.pdp = dict()
        self.summary_plot = dict()
        self.feature_impact = dict()
        self.multi_level_eda = dict()

    def create_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    def random_string_generator(self):
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        return random_str

    def caching_data_exist_in_dict(self, sql_query, graph_type):
        if graph_type == 'feature_importance':
            if sql_query in self.feature_importance:
                return True, self.feature_importance[sql_query]
            return False, None
        elif graph_type == 'pdp':
            if sql_query in self.pdp:
                return True, self.pdp[sql_query]
            return False, None
        elif graph_type == 'summary_plot':
            if sql_query in self.summary_plot:
                return True, self.summary_plot[sql_query]
            return False, None
        elif graph_type == 'feature_impact':
            if sql_query in self.feature_impact:
                return True, self.feature_impact[sql_query]
            return False, None
        elif graph_type == 'multi_level_eda':
            if sql_query in self.multi_level_eda:
                return True, self.multi_level_eda[sql_query]
            return False, None
        elif graph_type == 'filtered_df':
            if sql_query in self.query_dict:
                return True, self.query_dict[sql_query]
            return False, None

    def caching_exists_in_file(self, sql_query, graph_type):
        self.create_dir("data_storage")
        self.create_dir("data_storage/{}".format(graph_type))

        try:
            dictionary_csv = pd.read_csv("./data_storage/filtered_df/dictionary_bkp.csv")
            random_id = dictionary_csv[(dictionary_csv['sql'] == sql_query) & (dictionary_csv['type'] == graph_type)
                                       & (dictionary_csv['instance_id'] == self.instance_id)]

            random_id.drop_duplicates(['sql'], keep='last', inplace=True)
            if not random_id.empty:
                result = random_id['random_id'].iloc[0]
                dff = pd.read_csv("./data_storage/{}/{}.csv".format(graph_type, result))
                #print("{}  {}  {}".format(graph_type, "file exists", result))
                return True, True, dff
            return False, True, None

        except Exception as e:
            return False, False, None

    def creating_filtered_backup_file(self, sql_query, random_str, graph_type):
        dict_bkp = pd.DataFrame(data={"sql": [sql_query], "random_id": [random_str], "type": [graph_type],
                                      "instance_id": [self.instance_id]})
        return dict_bkp

    def store_data_in_csv(self, df, graph_type, random_str):
        df.to_csv("./data_storage/{}/{}.csv".format(graph_type, random_str), index=False)

    def store_data_in_dict(self, df, sql_query, graph_type):
        if graph_type == 'feature_importance':
            self.feature_importance[sql_query] = df
        elif graph_type == 'pdp':
            self.pdp[sql_query] = df
        elif graph_type == 'summary_plot':
            self.summary_plot[sql_query] = df
        elif graph_type == 'feature_impact':
            self.feature_impact[sql_query] = df
        elif graph_type == 'multi_level_eda':
            self.multi_level_eda[sql_query] = df
        elif graph_type == 'filtered_df':
            self.query_dict[sql_query] = df

    def caching_data_manager(self, df, sql_query, graph_type, calculated_funct=None, details_dict=None):
        status_file, file_exist, dff = self.caching_exists_in_file(sql_query, graph_type)
        if status_file:
            #print("{}/{}".format(graph_type, "exist in file"))

            return dff
        else:
            #print("{}/{}".format(graph_type, "don't exists"))
            random_str = self.random_string_generator()
            dict_bkp = self.creating_filtered_backup_file(sql_query, random_str, graph_type)
            dff = psql.sqldf(sql_query, locals())
            self.create_dir("data_storage/filtered_df")
            dict_bkp.to_csv("./data_storage/{}/dictionary_bkp.csv".format("filtered_df"), mode='a',
                            header=file_exist is False,
                            index=False)
            dff.to_csv("./data_storage/filtered_df/{}.csv".format(random_str), mode='w',
                       header=file_exist is False,
                       index=False)

            results = dff
            if graph_type != 'filtered_df':
                results = calculated_funct(dff)
            self.store_data_in_csv(results, graph_type, random_str)
            return results

    def find(self, df, y_variable, y_variable_predict, mode, param):
        self.available_columns = available_columns = list(df.columns)
        original_variables = [col for col in df.columns if '_impact' in col]
        self.impact_variables = [col for col in original_variables if not '_rescaled' in col]
        self.y_variable = y_variable
        self.y_variable_predict = y_variable_predict
        self.param = param
        self.instance_id = self.random_string_generator()
        self.create_dir("data_storage")
        self.create_dir("data_storage/user")
        self.user_id = None
        try:
            user_id = pd.read_csv("data_storage/user/user_id.csv")
            user_id.drop_duplicates(['id'], keep='first', inplace=True)
            user_id = user_id['id'].iloc[0]
            self.user_id = user_id
        except Exception as e:
            #print("inside user track" )
            user_id_val = self.random_string_generator()
            user_id_csv = pd.DataFrame(data={"id": [user_id_val]})
            user_id_csv.to_csv("data_storage/user/user_id.csv", index=False)
            self.user_id= user_id_val


            
        self.insights = insights(self.param)
        d = self.dash(df, mode)
        return True

    def dash(self, df, mode):

        y_variable = self.y_variable

        g = plotly_graphs()
        y_variables = [col for col in df.columns if '_impact' in col]
        original_variables = [col for col in df.columns if not '_impact' in col]
        original_variables = [col for col in original_variables if not '_rescaled' in col]
        array = []
        PAGE_SIZE = 10

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
                    href="https://explainx.ai",
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
                    href="https://explainx.ai",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),

            ],
            color="dark",
            dark=True,
        )

        app.layout = html.Div([
            navbar,
            html.Div([
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "Analyze using SQL",
                                    id="collapse-button-2",
                                    color="link",
                                    style={'fontSize': '14px'})
                            )
                        ),
                        dbc.Collapse(html.Div([
                            html.Div(dcc.Input(
                                id='input-on-submit',
                                type='text',
                                placeholder="SELECT * FROM df",
                                value="SELECT * FROM df",
                                style={'height': '200px', 'width': '700px', 'fontSize': '15px'})),

                            html.Button('Execute Query', id='submit-val', n_clicks=0),
                            html.Div(id='sql-query-button',
                                     children='Enter a value and press submit',
                                     style={'display': 'none'}
                                     )

                        ], style={'marginTop': 0}), id="collapse-2"),
                    ]
                ),

            ], style=style4),

            html.Div([
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H2(
                                dbc.Button(
                                    "View Your Data",
                                    id="collapse-button",
                                    color="link",
                                    style={'fontSize': '14px'})
                            )
                        ),
                        dbc.Collapse(html.Div([
                            html.H4('',
                                    style=style1),
                            html.Div([
                                dash_table.DataTable(
                                    id='datatable-interactivity',
                                    columns=[
                                        {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
                                    ],
                                    # data=df.to_dict('records'),
                                    editable=True,

                                    sort_mode="multi",

                                    # selected_columns=[],
                                    # selected_rows=[],
                                    # page_action="native",
                                    page_current=0,
                                    page_size=PAGE_SIZE,
                                    row_selectable='multi',
                                    page_action='custom',
                                    style_table=style2,
                                    style_header=style3,

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

                        ], style={'marginTop': 0}), id="collapse"),
                    ]
                ),

            ],
                style=style4),

            # end of collapsable div
            dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                dcc.Tab(label='Global Explanation', value='tab-1', children=[

                    # Global Feature Impact & Importance
                    html.Div(id='datatable-interactivity-container', children=[
                        html.Div([
                            html.Div([
                                html.H4('Global Feature Importance',
                                        style=style5),
                                html.P(
                                    'Feature importance assign a score to input features based on how useful they are at predicting a target variable. ',
                                    style=style6)
                                ,
                                dcc.Loading(
                                    id="loading-1",
                                    type="circle",
                                    children=dbc.Row(
                                        [
                                            dbc.Col(html.Div(dcc.Graph(id="global_feature_importance",
                                                                       style={'marginLeft': 50, 'marginTop': 0,
                                                                              'height': '500px'}
                                                                       )), width=9),
                                            dbc.Col(
                                                [
                                                    html.Div([
                                                        html.H2("How to read this graph?"),
                                                        html.P(
                                                            "This graph helps you identify which features in your dataset have the greatest effect on the outcomes of your machine learning model")
                                                    ]),
                                                    html.Div([
                                                        html.H2("Insights"),
                                                        html.P(id='global_message_1'),
                                                        html.P(id='global_message_2'),
                                                        html.P(id='global_message_3'),
                                                        html.P(id='global_message_4'),
                                                        html.P(id='global_message_5')

                                                    ]

                                                    )]
                                            )

                                        ])

                                )
                                , ], style=style7,
                            ),

                            html.Div([
                                html.H4('Global Feature Impact',
                                        style=style8),
                                html.P(
                                    'Feature impact identifies which features (also known as columns or inputs) in a dataset have the greatest positive or negative effect on the outcomes of a machine learning model.',
                                    style=style9),
                                dcc.Loading(
                                    id="loading-2",
                                    type="circle",
                                    children=dbc.Row(
                                        [
                                            dbc.Col(html.Div(dcc.Graph(id='global_feature_impact',
                                                                       style={'marginLeft': 50, 'marginTop': 0,
                                                                              'height': '500px'})), width=9),
                                            dbc.Col(
                                                [
                                                    html.Div([
                                                        html.H2("How to read this graph?"),
                                                        html.P(
                                                            "This tells you which features have positive impact and which features have negative impact on the output of the decision")
                                                    ]),
                                                    html.Div([
                                                        html.H2("Insights"),
                                                        html.P(id='message_1'),
                                                        html.P(id='message_2'),
                                                        html.P(id='message_3')
                                                    ]

                                                    )]
                                            )

                                        ])

                                )

                            ],
                                style=style10,
                            ),
                        ],

                            style=style11
                        )
                    ], style={'height': '400'})
                ]),

                dcc.Tab(label='Feature Interaction', value='tab-2', children=[

                    dcc.Loading(
                        id="feature_interaction_load",
                        type="circle",
                        children=html.Div([
                            html.Div([
                                html.H4('Partial Dependence Plot',
                                        style=style12),
                                html.P(
                                    'The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model',
                                    style=style13),
                                html.Div([
                                    html.P('Variable Name'),
                                    dcc.Dropdown(
                                        id='xaxis-column',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[1],
                                        clearable=False
                                    ),

                                ],
                                    style={'width': '20%', 'marginLeft': 70, 'float': 'left',
                                           'display': 'inline-block'}),

                                html.Div([
                                    html.P("Variable Impact Values"),
                                    dcc.Dropdown(
                                        id='yaxis-column',
                                        options=[{'label': i, 'value': i} for i in y_variables],
                                        value=y_variables[1],
                                        clearable=False
                                    ),

                                ], style=style14),

                                html.Div([
                                    html.P('3rd Variable'),
                                    dcc.Dropdown(
                                        id='third-axis',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[-3],
                                        clearable=False
                                    ),

                                ], style=style15),

                            ]),
                            dcc.Loading(
                                id="loading-5",
                                type="circle",
                                children=dcc.Graph(id='indicator-graphic', style={'marginLeft': 50})
                            ),
                        ],
                            style=style16),
                    ),

                    dcc.Loading(
                        id="loading-2-pdp",
                        type='circle',
                        children=html.Div([
                            html.Div([
                                html.H4('Summary Plot',
                                        style=style17),
                                html.P(
                                    'In the summary plot, we see first indications of the relationship between the value of a feature and the impact on the prediction',
                                    style=style18)
                                ,
                                dcc.Loading(
                                    id="loading-3",
                                    type="circle",
                                    children=dcc.Graph(id='summary_plot', style={'marginLeft': 50, 'height': '600px'})

                                ),

                            ], style=style19),
                        ], style=style20)
                    )

                ]),

                # tab 3:Distribution

                dcc.Tab(label='Distribution', value='tab-3', children=[

                    html.Div([

                        html.Div([
                            html.Div([
                                html.H4('Distributions',
                                        style=style21),
                                html.Div([
                                    html.P('Variable Name'),
                                    dcc.Dropdown(
                                        id='xaxis-column-name',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[0]
                                    ),
                                ],
                                    style=style22),
                                html.Div([
                                    html.P('Plot Type'),
                                    dcc.Dropdown(
                                        id='plot_type',
                                        options=[{'label': 'Violin Plot', 'value': 'Violin Plot'},
                                                 {'label': 'Histogram', 'value': 'Histogram'}, ],
                                        value='Histogram'
                                    ),
                                ],
                                    style=style23),

                            ]),

                            dcc.Graph(id='indicator-graphic2',
                                      style={'marginLeft': 50, 'marginTop': 100, 'height': '500px'}),
                        ],
                            style=style24),

                        html.Div([

                            html.Div([
                                html.H4('Multi-Level EDA', style=style21),
                                html.Div([
                                    html.P('X-Axis Variable'),
                                    dcc.Dropdown(
                                        id='x_axis',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[0]
                                    ),
                                ], style=style22),
                                html.Div([
                                    html.P('Y-Axis Variable'),
                                    dcc.Dropdown(
                                        id='y_axis',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[1]
                                    ),
                                ], style=style22),
                                html.Div([
                                    html.P('Size'),
                                    dcc.Dropdown(
                                        id='size',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        # value=original_variables[0]
                                        placeholder="Choose a variable"
                                    ),
                                ], style=style22),
                                html.Div([
                                    html.P('Color'),
                                    dcc.Dropdown(
                                        id='color',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        placeholder="Choose a variable"
                                    ),
                                ], style=style22),
                                html.Div([
                                    html.P('Drill-down by Column'),
                                    dcc.Dropdown(
                                        id='facet_col',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        placeholder="Select a categorical variable"
                                        # value=original_variables[0]
                                    ),
                                ], style=style22),
                                html.Div([
                                    html.P('Drill-down by Row'),
                                    dcc.Dropdown(
                                        id='facet_row',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        placeholder="Select a categorical variable"
                                    ),
                                ], style=style22)
                            ]),
                            dcc.Graph(id='multi_level_eda',
                                      style={'marginLeft': 50, 'marginTop': 200, 'height': '700px'})

                        ], style=style24)

                    ], style=style25)

                ]),
                # tab 4:Local Explanation

                dcc.Tab(label='Local Explanation', value='tab-4', children=[

                    html.Div([

                        # Input and Data
                        html.Div([

                            html.Div([
                                html.H4('Data',
                                        style=style26),
                                html.I("Index of Row That You Want To Explain.", style={'padding-left': '70px'}),

                                dcc.Input(id="row_number", type="number", value=1,
                                          placeholder="Enter a Row Number e.g. 1, 4, 5",
                                          style={'text-align': 'center'}),

                                html.Div(id='local_data_table', style={'marginTop': "20px", 'marginBottom': "5px"}),

                            ]),

                        ]),  # End of Input & Data Div

                        dcc.Tabs([
                            dcc.Tab(label='Local Feature Explanation', children=[

                                html.Div(id='datatable-interactivity-container2', children=[
                                    html.Div([

                                        html.Div([
                                            html.H4('Local Feature Impact',
                                                    style=style29),
                                            html.P(
                                                'Local Feature impact identifies which features have the greatest positive or negative effect on the outcome of a machine learning model for a specific row.',
                                                style=style30),
                                            dcc.Loading(
                                                id="local_feature_impact_1",
                                                type="circle",
                                                children=dbc.Row(
                                                    [
                                                        dbc.Col(html.Div(dcc.Graph(id='local_feature_impact',
                                                                                   style={'marginLeft': 50,
                                                                                          'marginTop': 0,
                                                                                          'height': '500px'})),
                                                                width=8),
                                                        dbc.Col(
                                                            [
                                                                html.Div([
                                                                    html.H2("How to read this graph?"),
                                                                    html.P(
                                                                        "According to the model, the features are most important in explaining the target variable. Most importance is on the top.")
                                                                ]),
                                                                html.Div([
                                                                    html.H2("Insights"),
                                                                    html.P(id='local_message_1'),
                                                                    html.P(id='local_message_2'),
                                                                    html.P(id='local_message_3')
                                                                ]),
                                                                html.Div([
                                                                    html.H2("Next Steps"),
                                                                    html.P(
                                                                        "Click on the similar profile tabs and explore profiles that had similar attributes"),
                                                                ])
                                                            ]
                                                            , width=4),

                                                    ]))

                                        ],
                                            style=style31,
                                        ),
                                    ],

                                        style=style32)], style={'height': '400'})

                            ]),

                            dcc.Tab(id="similar_tabs", label='Similar Profiles', children=[

                                html.Div([
                                    html.H4('Similar Profiles',
                                            style=style33),

                                    html.Button('Display Similar Profiles', id='btn-nclicks-1', n_clicks=0,
                                                style=style34),

                                    html.P(
                                        'Display similar  profiles and the extent to which they are similar to the chosen applicant as indicated by the last row in the table below labelled as "Weight"',
                                        style=style35),
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
                                                style_table=style36,
                                                style_header=style37,
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

                                                    },

                                                    {

                                                        'if': {

                                                            'filter_query': '{0_color} =1',

                                                            'column_id': '0'

                                                        },

                                                        'backgroundColor': 'tomato',

                                                        'color': 'white'

                                                    },

                                                    {

                                                        'if': {

                                                            'filter_query': '{1_color} =1',

                                                            'column_id': '1'

                                                        },

                                                        'backgroundColor': 'tomato',

                                                        'color': 'white'

                                                    },

                                                    {

                                                        'if': {

                                                            'filter_query': '{2_color} =1',

                                                            'column_id': '2'

                                                        },

                                                        'backgroundColor': 'tomato',

                                                        'color': 'white'

                                                    },

                                                    {

                                                        'if': {

                                                            'filter_query': '{3_color} =1',

                                                            'column_id': '3'

                                                        },

                                                        'backgroundColor': 'tomato',

                                                        'color': 'white'

                                                    },

                                                    {

                                                        'if': {

                                                            'filter_query': '{4_color} =1',

                                                            'column_id': '4'

                                                        },

                                                        'backgroundColor': 'tomato',

                                                        'color': 'white'

                                                    }

                                                ],
                                                #
                                            )
                                        )

                                    ])

                                ], style=style39),

                            ])

                        ], style={'marginTop': 50}), ])

                ])

            ]), ], style=style40)

        @app.callback(
            Output('datatable-interactivity', 'data'),
            [Input('datatable-interactivity', "page_current"),
             Input('datatable-interactivity', "page_size")])
        def update_table(page_current, page_size):
            return df.iloc[
                   page_current * page_size:(page_current + 1) * page_size
                   ].to_dict('records')

        @app.callback(
            Output("collapse", "is_open"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("collapse-2", "is_open"),
            [Input("collapse-button-2", "n_clicks")],
            [State("collapse-2", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            dash.dependencies.Output('sql-query-button', 'children'),
            [dash.dependencies.Input('submit-val', 'n_clicks')],
            [dash.dependencies.State('input-on-submit', 'value')])
        def update_output(n_clicks, value):
            # sql_query = f'SELECT * FROM df {value}'
            sql_query = f'{value}'

            return sql_query

        # Global Feature Importance
        @app.callback(
            [Output('global_feature_importance', "figure"),
             Output('global_message_1', "children"),
             Output('global_message_2', "children"),
             Output('global_message_3', "children"),
             Output('global_message_4', "children"),
             Output('global_message_5', "children")],
            [Input('sql-query-button', 'children'),
             Input('tabs-styled-with-inline', 'value')])
        def update_graphs(sql_query, tab):
            if tab == "tab-1":
                g = plotly_graphs()
                graph_type = "feature_importance"
                dff = self.caching_data_manager(df, sql_query, graph_type, g.feature_importance)
                figure = g.feature_importance_graph(dff)
                message = self.insights.insight_1_feature_imp(dff)
                if len(message) == 4:
                    return figure, message[0], message[1], message[2], message[3], ""
                return figure, message[0], message[1], message[2], message[3], message[4]
            else:
                return {}, '', '', '', '', ''

        # Global Feature Impact
        @app.callback(
            [Output('global_feature_impact', "figure"),
             Output('message_1', "children"),
             Output('message_2', "children"),
             Output('message_3', "children")],
            [Input('sql-query-button', 'children'),
             Input('tabs-styled-with-inline', 'value')])
        def update_graphs(sql_query, tab):
            if tab == 'tab-1':
                g = plotly_graphs()
                graph_type = "feature_impact"
                dff = self.caching_data_manager(df, sql_query, graph_type, g.feature_impact)
                figure = g.feature_impact_graph(dff)
                message = self.insights.insight_2_global_feature_impact(dff)
                return figure, message[0], message[1], message[2]
            else:
                return {}, '', '', ''

        @app.callback(
            Output(component_id='local_data_table', component_property='children'),
            [Input(component_id='row_number', component_property='value')])
        def update_local_table(row_number):
            i = 0
            if type(row_number) == type(1):
                i = row_number
            array = df[i:i + 1]
            array1 = array
            impact_variables = [col for col in array if '_impact' in col]
            for col in impact_variables:
                array1.drop([col], axis=1, inplace=True)
            figure = dbc.Table.from_dataframe(array1, striped=True, bordered=True, hover=True, responsive=True,
                                              size='lg', style={'font-size': '12px'})
            return figure

        # Local Feature Impact Graph
        @app.callback(
            [Output('local_feature_impact', "figure"),
             Output('local_message_1', "children"),
             Output('local_message_2', "children"),
             Output('local_message_3', "children")],
            [Input(component_id='row_number', component_property='value'),
             Input('tabs-styled-with-inline', 'value')])
        def update_impact_graph(row_number, tab):
            if tab == 'tab-4':
                i = 0
                if type(row_number) == type(1):
                    i = row_number
                array = df[i:i + 1]
                figure, dat = g.feature_impact_old(array)
                message = self.insights.insight_2_local_feature_impact(dat)
                return figure, message[0], message[1], message[2]
            else:
                return {}, '', '', ''

        # Partial Dependence Plot Graph
        @app.callback(
            Output('indicator-graphic', 'figure'),
            [Input('xaxis-column', 'value'),
             Input('yaxis-column', 'value'),
             Input('third-axis', 'value'),
             Input('sql-query-button', 'children'),
             Input('tabs-styled-with-inline', 'value')])
        def update_graph(xaxis_column_name, yaxis_column_name, third_axis_name, sql_query, tab):
            if tab == 'tab-2':
                g = plotly_graphs()
                graph_type = 'pdp'
                df3 = self.caching_data_manager(df, sql_query, graph_type, g.partial_dependence_plot)
                fig = g.pdp_plot(df3, df3[xaxis_column_name], df3[yaxis_column_name],
                                 df3[third_axis_name])
                return fig
            else:
                return {}

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

                dfs, sample_row = p.find_prototypes(row_number)

                dat = dfs.T.reset_index()

                print("sample row columns")

                sample_row = sample_row.to_frame()

                sample_row.rename(columns={sample_row.columns[0]: "orig"}, inplace=True)

                sample_row.reset_index(inplace=True)

                print(sample_row.columns)

                dat = pd.merge(dat, sample_row, on=['index'], how='left')

                dat['orig'] = dat['orig'].astype(float)

                for i in list(dat.columns):

                    dat[str(i) + '_color'] = np.nan

                    if i != 'index':

                        dat[i] = dat[i].astype(float)

                        dat[str(i) + '_color'] = dat[str(i) + '_color'].astype(float)


                    dat[str(i) + '_color'] = np.where(dat['orig'] == dat[i], 1, 0)


                dat.drop(["index_color","orig_color"], axis=1, inplace=True)

                dat = dat.to_dict('records')

                return dat

            else:
                return []

        # Summary Plot
        @app.callback(
            Output('summary_plot', 'figure'),
            [Input('sql-query-button', 'children'),
             Input('tabs-styled-with-inline', 'value')])
        def update_graph2(sql_query, tab):
            if tab == 'tab-2':
                graph_type = 'summary_plot'
                g = plotly_graphs()
                dff = self.caching_data_manager(df, sql_query, graph_type, g.summary_plot)
                fig = g.summary_plot_graph(dff)
                return fig
            else:
                return {}

        # Distributions
        @app.callback(
            Output('indicator-graphic2', 'figure'),
            [Input('xaxis-column-name', 'value'),
             Input('plot_type', 'value'),
             Input('sql-query-button', 'children'),

             ])
        def update_graph2(xaxis_column_name, plot_type, sql_query):

            graph_type = 'filtered_df'
            df3 = self.caching_data_manager(df, sql_query, graph_type)
            cat_variables = []
            num_variables = []
            for i in original_variables:
                if df[i].dtype == 'object':
                    cat_variables.append(i)
                else:
                    num_variables.append(i)

            if plot_type == "Histogram":
                # group_labels = ['xxaxis_column_name']
                return px.histogram(df3, x=xaxis_column_name, marginal="box")
            else:
                for i in cat_variables:
                    return px.violin(df3, x=xaxis_column_name, box=True, points='all')
                else:
                    return px.violin(df3, y=xaxis_column_name, box=True, points='all', )

        # Multi Level EDA
        @app.callback(
            Output('multi_level_eda', 'figure'),
            [Input('x_axis', 'value'),
             Input('y_axis', 'value'),
             Input('size', 'value'),
             Input('color', 'value'),
             Input('facet_col', 'value'),
             Input('facet_row', 'value'),
             Input('sql-query-button', 'children'),
             ]
        )
        def multi_level(x_axis, y_axis, size, color, facet_col, facet_row, sql_query):
            graph_type = 'filtered_df'
            df3 = self.caching_data_manager(df, sql_query, graph_type)
            return px.scatter(df3, x=x_axis, y=y_axis, size=size, color=color, facet_col=facet_col, facet_row=facet_row,
                              facet_col_wrap=4)

        # Port Finder
        port =8080
        if mode == "inline":
            try:
                app.run_server(mode="inline", port=port)
            except:
                port= self.find_free_port()
                app.run_server(mode="inline",port=port)
        else:
            try:
                app.run_server(host='0.0.0.0', port=port)

            except:
                # try different ip in case 0.0.0.0 does not work
                try:
                    try:
                        port=self.find_free_port()
                        app.run_server(host='0.0.0.0', port=port)
                    except:
                        port = self.find_free_port()
                        app.run_server(host='0.0.0.0', port=port)
                except:
                    try:
                        port = self.find_free_port()
                        app.run_server(host='127.0.0.1', port=port)
                    except:
                        print("Please restart Jupyter Notebook or Python IDE.")
                        return False
        #
        # localtunnel(port)

        #update counter here
        try:
            self.increate_counter()
        except:
            pass


        return True

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def increate_counter(self):
        # call api call here

        url = 'https://us-central1-explainx-25b88.cloudfunctions.net/increaseCounter'
        params = {

        }
        data = {
            "user_id": self.user_id,
            "model": self.param["model_name"]

        }
        r = requests.post(url, params=params, json=data)
        if r.json()["message"]=="200":
            return True
        else:
            return False


def localtunnel(port):
    # subdomain= 'explainx-'+ get_random_string(10)
    task = subprocess.Popen(['lt', '-h', '"https://serverless.social"', '-p', str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    outpt= task.stdout.readline()
    outpt_string= outpt.decode("utf-8").split("is:")
    print('Explainx.ai is running @ '+outpt_string[1])


def get_random_string(length):
    letters = string.ascii_lowercase+ string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

