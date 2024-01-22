from imports import *
from plotly_graphs import *
from protodash import *
from insights import *
from plotly_css import *
import pandasql as psql
import string
import random
import os
from apps import global_explanation, local_explanation, distribution, feature_interaction, cohort
from app import app
from what_if import *
from calculate_shap import *
from analytics import Analytics
from cohort_analysis import *


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
                # print("{}  {}  {}".format(graph_type, "file exists", result))
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
            # print("{}/{}".format(graph_type, "exist in file"))

            return dff
        else:
            # print("{}/{}".format(graph_type, "don't exists"))
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
                results = calculated_funct(dff, self.param["is_classification"])
            self.store_data_in_csv(results, graph_type, random_str)
            return results

    def find(self, df, mode, param):
        self.available_columns = available_columns = list(df.columns)
        original_variables = [col for col in df.columns if '_impact' in col]
        self.impact_variables = [col for col in original_variables if not '_rescaled' in col]
        self.y_variable = param["y_variable"]
        self.y_variable_predict = param["y_variable_predict"]
        self.param = param
        self.instance_id = self.random_string_generator()
        self.create_dir("data_storage")
        self.create_dir("data_storage/user")
        self.user_id = None
        self.df = df

        self.analytics = Analytics()
        self.analytics['ip'] = self.analytics.finding_ip()
        self.analytics['mac'] = self.analytics.finding_address()
        self.analytics['instance_id'] = self.instance_id
        self.analytics['time'] = str(datetime.datetime.now())
        self.analytics['total_columns'] = len(self.available_columns)
        self.analytics['total_rows'] = len(self.df)
        self.analytics['os'] = self.analytics.finding_system()
        self.analytics['model_name'] = self.param["model_name"]
        self.analytics["function"] = 'explainx.ai'
        self.analytics["query"] = "all"
        self.analytics['finish_time'] = ''

        self.callback_input = [Input(f + '_slider', 'value') for f in self.param["columns"]]
        self.callback_input.append(Input('submit-button-state', 'n_clicks'))

        # self.callback_input_prototype = [Input(f + '-slider', 'value') for f in self.param["columns"]]
        # self.callback_input_prototype.append(Input('btn-nclicks-1', 'n_clicks'))

        self.prototype_array = []
        for f in self.param["columns"]:
            self.prototype_array.append([f + '_slider', 'value'])
        self.prototype_array.append(['btn-nclicks-1', 'n_clicks'])
        try:
            user_id = pd.read_csv("data_storage/user/user_id.csv")
            user_id.drop_duplicates(['id'], keep='first', inplace=True)
            user_id = user_id['id'].iloc[0]
            self.user_id = user_id
        except Exception as e:
            # print("inside user track" )
            user_id_val = self.random_string_generator()
            user_id_csv = pd.DataFrame(data={"id": [user_id_val]})
            user_id_csv.to_csv("data_storage/user/user_id.csv", index=False)
            self.user_id = user_id_val

        self.analytics['user_id'] = self.user_id
        self.analytics.insert_data()
        self.insights = insights(self.param)
        d = self.dash(df, mode)
        return True

    def dash(self, df, mode):
        y_variable = self.y_variable
        g = plotly_graphs()
        ca = cohortAnalysis()
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

        app.title = "explainX.ai - Main Dashboard"

        PLOTLY_LOGO = "https://i.ibb.co/ZTWtVDV/explainx-logo.png"

        menu = dbc.Row(
            [   
                dbc.Col(dbc.NavItem(dbc.NavLink("Home", href="/apps/")),
                        style={'width': "150px", 'fontSize': '12px'}),
                dbc.Col(dbc.NavItem(dbc.NavLink("Global Explanation", href="/apps/global_explanation")),
                        style={'width': "150px", 'fontSize': '12px'}),
                dbc.Col(dbc.NavItem(dbc.NavLink("Local Explanation", href="/apps/local_explanation")),
                        style={'width': "150px", 'fontSize': '12px'}),
                dbc.Col(dbc.NavItem(dbc.NavLink("Feature Interaction", href="/apps/feature_interaction")),
                        style={'width': "150px", 'fontSize': '12px'}),
                dbc.Col(dbc.NavItem(dbc.NavLink("Distributions", href="/apps/distribution")),
                        style={'width': "150px", 'fontSize': '12px'}),
                dbc.Col(dbc.NavItem(dbc.NavLink("Cohort Analysis", href="/apps/cohort")),
                        style={'width': "150px", 'fontSize': '12px'})

            ],

            no_gutters=True,
            className="ml-auto flex-nowrap mt-3 mt-md-0",
            align="center"
        )

        navbar = dbc.Navbar(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                            dbc.Col(
                                dbc.NavbarBrand("explainX.ai",
                                            style={'fontSize': '15px', 'color': 'black'}
                                            )),
                            dbc.Col(
                                dbc.Button("Explain your own model!",
                                    color="danger",
                                    style={
                                        'fontSize': '12px'}))
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="https://www.github.com/explainx/explainx",
                ),

                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(menu, id="navbar-collapse", navbar=True),
            ],
            color="light",
            dark=True,
        )

        # add callback for toggling the collapse on small screens
        @app.callback(
            Output("navbar-collapse", "is_open"),
            [Input("navbar-toggler", "n_clicks")],
            [State("navbar-collapse", "is_open")],
        )
        def toggle_navbar_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(Output('page-content', 'children'),
                      [Input('url', 'pathname')])
        def display_page(pathname):
            if pathname == '/apps/global_explanation':
                return global_explanation.global_explanation(original_variables)
            elif pathname == '/apps/feature_interaction':
                return feature_interaction.layout_interaction(original_variables, y_variables)
            elif pathname == '/apps/distribution':
                return distribution.layout_distribution(original_variables)
            elif pathname == '/apps/local_explanation':
                return local_explanation.layout_local(original_variables, columns, df.columns)
            elif pathname == '/apps/cohort':
                return cohort.cohort_layout(original_variables)
            else:
                return welcome_message

        welcome_message = html.Div(
            [
                html.Div([
                    html.H3("Welcome to ExplainX.ai Explainable AI Dashboard"),
                    html.H4("Start explaining your model by exploring one of the following options.")
                ], style={'margin-left': "20px"}),
                html.Div([
                    html.Div([
                        html.Img(
                            src='https://lh3.googleusercontent.com/uNkpoLQRIza6SSDk9fQed6bu6c1Q9zDkDjZqkRtF3cU97Smf2rS0soKo2ZQm32gJe8FyLYvNRUYJ-hqM7zs4esIAFchdPmgC4sHgbpxTKkGjm4dRK-NagI56K8vnXg8FH4s_Jct2RaQ',
                            className="global_explanation_image"),

                        dcc.Link("Global Explanation", href="/apps/global_explanation", className="link")
                    ], className="welcome_box", id="global_explanation_deets"),

                    html.Div([
                        html.Img(
                            src='https://lh3.googleusercontent.com/LpM-N2QavBVxlMW6vKm5fmT5Qe3mYtI2q2OQc5kgZfW4kKOFlyidgrwT5C9zB3XHZncHvlGcFl82Si7bMDQN_PtLf4PThqdEUs1UwoUPeNHs0VYY6ICKlHi44ppZLZIMQvaWI4bMqm0',
                            className="global_explanation_image"),

                        dcc.Link("Local Level Explanation", href="/apps/local_explanation", className="link")

                    ], className="welcome_box", id="local_explanation_deets"),

                    html.Div([
                        html.Img(
                            src='https://lh6.googleusercontent.com/GXnNtGlqQcsRXugwwlZ31SwrJda4Z1WVyIi5i13vZImrUY6YgIZvM81d7zrBOh8n2jK_Lqj3BB5IqS4dU5l680-rZuu40bMXa-EAd8ag3WDIU2SNYieg0DH3e_FwUq5f2-y-X_POE-k',
                            className="global_explanation_image"),

                        dcc.Link("Feature Interactions", href="/apps/feature_interaction", className="link")

                    ], className="welcome_box", id="feature_interaction_deets"),

                    html.Div([
                        html.Img(
                            src='https://uploads-ssl.webflow.com/5edc18e72f7ba21e1ed2efae/5f519fd99580494301e67b7b_pipeline.png',
                            className="global_explanation_image"),

                        dcc.Link("Cohort Analysis", href="/apps/cohort", className="link")

                    ], className="welcome_box", id="cohort_analysis_deets")
                ], className="main_welcome_div"),

            ]

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
                                    style={'fontSize': '10px', 'margin-top': '0px'}), style={"margin-top": "0px"}),
                            style={"height": "50px"}),
                        dbc.Collapse(html.Div([
                            html.Div(dcc.Input(
                                id='input-on-submit',
                                type='text',
                                # placeholder="SELECT * FROM df"
                                value="SELECT * FROM df",
                                style={'height': '200px', 'width': '700px', 'fontSize': '10px'})),

                            html.Button('Execute Query', id='submit-val', n_clicks=1),
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
                                    style={'fontSize': '10px'}), style={"margin-top": "0px"}),
                            style={"height": "50px"}),
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
            # Pages Data
            html.Div([
                dcc.Location(id='url', refresh=False),
                html.Div(id='page-content')
            ])
            # end of collapsable div
        ], className="main_div")

        # Navigation

        # Data Interactivity
        @app.callback(
            Output('datatable-interactivity', 'data'),
            [Input('datatable-interactivity', "page_current"),
             Input('datatable-interactivity', "page_size")])
        def update_table(page_current, page_size):
            return df.iloc[
                   page_current * page_size:(page_current + 1) * page_size
                   ].to_dict('records')

        # Collapse-Toggle
        @app.callback(
            Output("collapse", "is_open"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        # Collapse-Toggle 2
        @app.callback(
            Output("collapse-2", "is_open"),
            [Input("collapse-button-2", "n_clicks")],
            [State("collapse-2", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        # Cohort Analysis - Callbacks
        @app.callback(
            Output("modal", "is_open"),
            [Input("open", "n_clicks"), Input("close", "n_clicks")],
            [State("modal", "is_open")],
            prevent_initial_call=True)
        def toggle_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            [Output("cohort-metrics-div", "children"),
             Output("cohort-details", "children"),
             Output("cohort-graph", "children")],
            [Input("add-cohort", "n_clicks"),
             Input("remove-cohort", "n_clicks"),
             Input("x-axis", "value")],
            [State("demo-dropdown", "value"),
             State("demo-operators", "value"),
             State("demo-values", "value")])
        def cohort_metrics_details(add_cohort, remove_cohort, x_axis, var_name, operator, value):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'remove-cohort' in changed_id:
                ca.remove_cohort()
                fig = ca.cohort_graph(x_axis)
                return ca.cohort_metrics_details(), ca.cohort_details(), dcc.Graph(figure=fig),
            else:
                ca.add_cohort_metrics(df, var_name, operator, value, self.param["is_classification"])
                cohort = ca.add_cohort(df, x_axis, var_name, operator, value)
                fig = ca.cohort_graph(x_axis)
                return ca.cohort_metrics_details(), ca.cohort_details(), dcc.Graph(figure=fig)

        # SQL - Data Input Callback
        @app.callback(
            dash.dependencies.Output('sql-query-button', 'children'),
            [dash.dependencies.Input('submit-val', 'n_clicks')],
            [dash.dependencies.State('input-on-submit', 'value')])
        def update_output(n_clicks, value):
            sql_query = f'{value}'
            return sql_query

        # What-If Form CallBack
        @app.callback(
            Output('place_form_here', 'children'),
            [Input('row_number', 'value')])
        def create_what_if_form(row_number):
            self.analytics.update_data()
            self.analytics['function'] = "what_if"
            self.analytics['time'] = str(datetime.datetime.now())
            self.analytics['query'] = row_number
            self.analytics.insert_data()
            self.analytics['finish_time'] = ''
            x = what_if()
            i = 0
            if type(row_number) == type(1):
                i = row_number
            array = df[i:i + 1]
            array1 = array
            impact_variables = [col for col in array if '_impact' in col]
            for col in impact_variables:
                array1.drop([col], axis=1, inplace=True)
            # features = [col for col in array if not '_impact' in col]
            features = list(self.param["columns"])
            features.append("y_prediction")
            features.append("y_actual")

            form = x.what_if_form(array1, features)
            return form

        """
        change this. Take input from what-if form.
        """

        # Local Feature Impact Graph
        @app.callback(
            [Output('local_feature_impact', "figure"),
             Output('local_message_1', "children"),
             Output('local_message_2', "children"),
             Output('local_message_3', "children"),
             Output('local_message_4', 'children')],
            self.callback_input, prevent_initial_call=True)
        def update_impact_graph(*values):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            df = pd.DataFrame([values[:-1]])
            df.columns = self.param["columns"]
            array = self.calculate_prediction_shap(df)

            # Y_Pred
            # Probability_
            if self.param["is_classification"]:
                y_and_prob = []
                y_and_prob.append(int(array["y_prediction"]))
                y_and_prob.append(round(float(array["Probability_" + str(int(array["y_prediction"]))]), 2))
            else:
                y_and_prob = []
                y_and_prob.append(round(float(array["y_prediction"]), 2))

            # figure, dat = g.feature_impact_old(array)
            figure, dat = g.local_feature_impact_graph(array)
            message = self.insights.insight_2_local_feature_impact(dat, y_and_prob)

            return figure, message[0], message[1], message[2], message[3]

        # Prototypical Analysis
        """
        Change this. Take input from what-if from
        """

        @app.callback(
            Output(component_id='prototype_data', component_property='data'),
            [Input(f[0], f[1]) for f in self.prototype_array])
        def update_table(*values):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'btn-nclicks-1' in changed_id:
                # get x variables and prediction column from the data

                df_row = pd.DataFrame([values[:-1]])
                df_row.columns = self.param["columns"]

                # find prediction and impact values first here please.
                df_row = self.calculate_prediction(df_row)
                df_selected = df[list(self.param["columns"]) + [self.param["y_variable_predict"]]]

                row_number = len(df_selected)
                if not isinstance(df_row, pd.DataFrame):
                    df_row = df_row.as_data_frame()
                df_selected.loc[row_number] = df_row.values[0]

                p = protodash()
                p.preprocess_data(df_selected, self.param["y_variable_predict"])

                dfs, sample_row = p.find_prototypes(row_number)
                dat = dfs.T.reset_index()
                print("sample row columns")
                sample_row = sample_row.to_frame()
                sample_row.rename(columns={sample_row.columns[0]: "orig"}, inplace=True)
                sample_row.reset_index(inplace=True)
                # print(sample_row.columns)
                dat = pd.merge(dat, sample_row, on=['index'], how='left')
                dat['orig'] = dat['orig'].astype(float)
                for i in list(dat.columns):
                    dat[str(i) + '_color'] = np.nan
                    if i != 'index':
                        dat[i] = dat[i].astype(float)
                        dat[str(i) + '_color'] = dat[str(i) + '_color'].astype(float)
                    dat[str(i) + '_color'] = np.where(dat['orig'] == dat[i], 1, 0)
                dat.drop(["index_color", "orig_color"], axis=1, inplace=True)
                dat = dat.to_dict('records')
                return dat
            else:
                return []

        # Global Feature Importance
        @app.callback(
            [Output('global_feature_importance', "figure"),
             Output('global_message_1', "children")],
            [Input('sql-query-button', 'children'),
             Input('xaxis-column-test-2', 'value')])
        def update_graphs(sql_query, value):
            self.analytics.update_data()
            self.analytics['function'] = "feature_importance"
            self.analytics['time'] = str(datetime.datetime.now())
            self.analytics['query'] = sql_query
            self.analytics['finish_time'] = ''
            self.analytics.insert_data()
            g = plotly_graphs()
            graph_type = "feature_importance"
            dff = self.caching_data_manager(df, sql_query, graph_type, g.feature_importance)
            message = self.insights.insight_1_feature_imp(dff)
            figure = g.global_feature_importance_graph(dff, self.param["is_classification"])
            return figure, message[0]

        # Global Feature Impact
        @app.callback(
            [Output('global_feature_impact', "figure"),
             Output('message_1', "children"),
             Output('message_2', "children"),
             Output('message_3', "children")],
            [Input('sql-query-button', 'children'),
             Input('xaxis-column-test-2', 'value')])
        def update_graphs(sql_query, value):
            g = plotly_graphs()
            graph_type = "feature_impact"
            df3 = self.caching_data_manager(df, sql_query, graph_type, g.feature_impact)
            figure = g.global_feature_impact_graph(df3, self.param["is_classification"])
            message = self.insights.insight_2_global_feature_impact(df3)
            return figure, message[0], message[1], message[2]

        # Partial Dependence Plot Graph
        @app.callback(
            Output('indicator-graphic', 'figure'),
            [Input('xaxis-column', 'value'),
             #  Input('yaxis-column', 'value'),
             Input('third-axis', 'value'),
             Input('sql-query-button', 'children')])
        def update_graph(xaxis_column_name, third_axis_name, sql_query):
            self.analytics.update_data()
            self.analytics['function'] = "pdp"
            self.analytics['time'] = str(datetime.datetime.now())
            self.analytics['query'] = sql_query
            self.analytics.insert_data()

            g = plotly_graphs()
            graph_type = 'pdp'
            df3 = self.caching_data_manager(df, sql_query, graph_type, g.partial_dependence_plot)
            print(df3)
            fig = g.pdp_plot(df3, df3[xaxis_column_name], df3[xaxis_column_name + "_impact"], df3[third_axis_name])
            return fig

        # Summary Plot
        @app.callback(
            Output('summary_plot', 'figure'),
            [Input('sql-query-button', 'children'),
             Input('xaxis-column-test', 'value')])
        def update_graph2(sql_query, value):

            g = plotly_graphs()
            graph_type = 'summary_plot'
            df3 = self.caching_data_manager(df, sql_query, graph_type, g.summary_plot)
            fig = g.summary_plot_graph(df3)
            return fig

        # Distributions
        @app.callback(
            Output('indicator-graphic2', 'figure'),
            [Input('xaxis-column-name', 'value'),
             Input('plot_type', 'value'),
             Input('sql-query-button', 'children')])
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

                return px.histogram(df3, x=xaxis_column_name, marginal="box", template="plotly_white")
            else:
                for i in cat_variables:
                    return px.violin(df3, x=xaxis_column_name, box=True, points='all', template="plotly_white")
                else:
                    return px.violin(df3, y=xaxis_column_name, box=True, points='all', template="plotly_white")

        # Port Finder
        port = 8080
        debug_value = True

        if mode == "inline":
            try:
                app.run_server(mode="inline", port=port, debug=debug_value, dev_tools_ui=debug_value,
                               dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                               dev_tools_hot_reload=True)
            except:
                port = self.find_free_port()
                app.run_server(mode="inline", port=port, debug=debug_value, dev_tools_ui=debug_value,
                               dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                               dev_tools_hot_reload=True)
        else:
            try:
                app.run_server(host='127.0.0.1', port=port, debug=debug_value, dev_tools_ui=debug_value,
                               dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                               dev_tools_hot_reload=True)
            except:
                # try different ip in case 0.0.0.0 does not work
                try:
                    try:
                        port = self.find_free_port()
                        app.run_server(host='127.0.0.1', port=port, debug=debug_value, dev_tools_ui=debug_value,
                                       dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                       dev_tools_hot_reload=True)
                    except:
                        port = self.find_free_port()
                        app.run_server(host='127.0.0.1', port=port, debug=debug_value, dev_tools_ui=debug_value,
                                       dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                       dev_tools_hot_reload=True)
                except:
                    try:
                        port = self.find_free_port()
                        app.run_server(host='127.0.0.1', port=port, debug=debug_value, dev_tools_ui=debug_value,
                                       dev_tools_props_check=debug_value, dev_tools_silence_routes_logging=True,
                                       dev_tools_hot_reload=True)
                    except:
                        print("Please restart Jupyter Notebook or Python IDE.")
                        return False
        try:
            self.increate_counter()
        except:
            pass
        return port

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    # COUNTER FUNCTION (NEEDS TO BE IMPLEMENTED)
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
        if r.json()["message"] == "200":
            return True
        else:
            return False

    def calculate_prediction_shap(self, df):
        if self.param["model_name"] == "xgboost":
            import xgboost
            if xgboost.__version__ in ['1.1.0', '1.1.1', '1.1.0rc2', '1.1.0rc1']:
                print(
                    "Current Xgboost version is not supported. Please install Xgboost using 'pip install xgboost==1.0.2'")
                return False
            prediction_col = self.param["model"].predict(xgboost.DMatrix(df))

        elif self.param["model_name"] == "catboost":
            prediction_col = self.param["model"].predict(df.to_numpy())

        elif self.param['model_name'] == 'h2o':
            df = h2o.H2OFrame(df)
            prediction_col = self.param["model"].predict(df)

        else:
            prediction_col = self.param["model"].predict(df.to_numpy())

        # is classification?
        is_classification = self.param["is_classification"]

        # shap
        c = calculate_shap()
        df_final, explainer = c.find(self.param["model"], df, prediction_col, is_classification,
                                     model_name=self.param["model_name"])

        # prediction col
        # df_final["y_prediction"] = prediction_col

        if is_classification is True:

            try:
                df_final = self.formatting_y_pred_for_h2o_classification(df_final, prediction_col)
                # find and add probabilities in the dataset.
                prediction_col_prob = self.param["model"].predict_proba(df.to_numpy())
            except:
                prediction_col_prob = self.param["model"].predict(df)
                prediction_col_prob = prediction_col_prob.as_data_frame()
            pd_prediction_col_prob = pd.DataFrame(prediction_col_prob)

            for c in pd_prediction_col_prob.columns:
                df_final["Probability_" + str(c)] = list(pd_prediction_col_prob[c])

            # for c in pd_prediction_col_prob.columns:
            #     df_final["Probability_" + str(c)] = list(pd_prediction_col_prob[c])
            #     if c != 'predict':
            #         if "p" in c:
            #             res = c.split("p")[-1]
            #             df_final["Probability_" + str(res)] = list(pd_prediction_col_prob[c])
            #         else:
            #             df_final["Probability_" + str(c)] = list(pd_prediction_col_prob[c])
            #     else:
            #         df_final["Probability_" + str(c)] = list(pd_prediction_col_prob[c])
            df_final = self.formatting_h2o_prediction_prob(df_final, pd_prediction_col_prob)
        return df_final

    def formatting_y_pred_for_h2o_classification(self, final_df, pred_col):
        try:
            final_df["y_prediction"] = pred_col
        except:
            # df_final = df_final.as_data_frame()
            print("prediction col checking")
            prediction_col = pred_col.as_data_frame()
            final_df["y_prediction"] = prediction_col['predict'].iloc[0]
        return final_df

    def formatting_h2o_prediction_prob(self, final_df, h2o_pred):
        for c in h2o_pred.columns:
            final_df["Probability_" + str(c)] = list(h2o_pred[c])
            if c != 'predict':
                if "p" in c:
                    res = c.split("p")[-1]
                    final_df["Probability_" + str(res)] = list(h2o_pred[c])
                else:
                    final_df["Probability_" + str(c)] = list(h2o_pred[c])
            else:
                final_df["Probability_" + str(c)] = list(h2o_pred[c])
        return final_df

    def calculate_prediction(self, df):
        if self.param["model_name"] == "xgboost":
            import xgboost
            if xgboost.__version__ in ['1.1.0', '1.1.1', '1.1.0rc2', '1.1.0rc1']:
                print(
                    "Current Xgboost version is not supported. Please install Xgboost using 'pip install xgboost==1.0.2'")
                return False
            prediction_col = self.param["model"].predict(xgboost.DMatrix(df))

        elif self.param["model_name"] == "catboost":
            prediction_col = self.param["model"].predict(df.to_numpy())

        elif self.param['model_name'] == 'h2o':
            df = h2o.H2OFrame(df)
            prediction_col = self.param["model"].predict(df)

        else:
            prediction_col = self.param["model"].predict(df.to_numpy())

        # is classification?
        is_classification = self.param["is_classification"]

        # prediction col
        df[self.param["y_variable_predict"]] = prediction_col

        return df


def localtunnel(port):
    # subdomain= 'explainx-'+ get_random_string(10)
    task = subprocess.Popen(['lt', '-h', '"https://serverless.social"', '-p', str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    outpt = task.stdout.readline()
    outpt_string = outpt.decode("utf-8").split("is:")
    print('Explainx.ai is running @ ' + outpt_string[1])


def get_random_string(length):
    letters = string.ascii_lowercase + string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
