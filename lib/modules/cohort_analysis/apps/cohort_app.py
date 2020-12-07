from .cohort_class import cohortAnalysis
import dash_core_components as dcc
import dash_html_components as html
from explainx.lib.utils import is_classification
from explainx.lib.imports import *

def test_func(x_test, model, app):

    ca = cohortAnalysis(model)

    var_name_dropdown = html.Div([
        html.P("Choose Variable"),
        dcc.Dropdown(
            id='demo-dropdown',
            options=[{'label': i, 'value': i} for i in x_test.columns],
            value= "",
            clearable=False
        )
    ])


    operators_list = ["==","=",">","<",">=","<="]

    operators_dropdown = html.Div([
        html.P("Choose Operator"),
        dcc.Dropdown(id="demo-operators",
                    options=[{"label":i, "value":i} for i in operators_list],
                    value = "",
                    clearable=False
                    )
    ])

    value_input = html.Div([
        html.P("Enter Value"),
        html.Div(id="demo-test")
    ])

    def signal(is_classification):
        if is_classification(model) == True:
            return x_test.columns[-4:]
        else:
            return x_test.columns[-2:]

    x_axis_dropdown = html.Div([
        html.P("Choose X-Axis Variable"),
        dcc.Dropdown(id="x-axis",
                    options = [{"label":i, "value":i} for i in signal(is_classification)],
                    value = x_test.columns[-2],
                    clearable=False)
    ], style={"width":"30%", "padding-left":"20px"})


    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader("Cohort Analysis"),
                    dbc.ModalBody(
                        html.Div( 
                                [var_name_dropdown,
                                operators_dropdown,
                                value_input  
                                ], id="modal_body")
                    ),
                    dbc.ModalFooter([
                        dbc.Button("Add Cohort", id="add-cohort", n_clicks=3),
                        dbc.Button("Close", id="close", className="ml-auto")
                    ])],
                id="modal",
            ),
        ], id="modal-parent"
    )

    button = dbc.Button("Add Cohort", id="open")


    remove_button = dbc.Button("Remove Cohort", id="remove-cohort", style={"margin-left":"20px"})

    cohort_details = html.Div(id="cohort-details", children=[], style={"display":"flex"})

    cohort_metrics_div = html.Div(id="cohort-metrics-div", children = [], style={"display":"flex"})

    heading = html.H3("Evaluate Model Performance - Cohort Analysis", style={"padding-left":"20px", "padding-top":"20px"})
    details = html.P("Evaluate the performance of your model by exploring the distribution of your prediction value and the values of your model performance metrics. You can further investigate your model by looking at a comparative analysis of its performance across different cohorts or subgroups of your dataset. Select filters along y-value and x-value to cut across different dimensions.", style={"padding-left":'20px'})

    card = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Cohort Distribution", tab_id="tab-1"),
                        dbc.Tab(label="Dataset Explorer", tab_id="tab-2"),
                    ],
                    id="card-tabs",
                    card=True,
                    active_tab="tab-1",
                )
            ),
            dbc.CardBody(html.P(id="card-content", className="card-text")),
        ]
    )


    layout = html.Div(
        [
            heading,
            details,
            html.Div([button, remove_button],style={"padding":"20px", "display":"flex"}),
            cohort_details,
            cohort_metrics_div,
            modal,
            card,


        ], id="main"
    )

    

    @app.callback(
            Output("card-content", "children"), 
            [Input("card-tabs", "active_tab")]
    )
    def tab_content(active_tab):
        if active_tab == 'tab-1':
            div = html.Div([
                html.Div(x_axis_dropdown),
                html.Div(id="cohort-graph")], style={"display":"block"})
            return div
        else:
            return "This is tab {}".format(active_tab)

    @app.callback(
        Output("modal", "is_open"),
        [Input("open", "n_clicks"), Input("close", "n_clicks")],
        [State("modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open


    #new callback added.
    @app.callback(Output('demo-test', "children"),
                [Input('demo-dropdown','value')])
    def categorical_support(value):
        is_cat = np.array([dt.kind == "O" for dt in x_test.dtypes])
        category_cols = x_test.columns.values[is_cat]
        if value in category_cols:
            return dcc.Dropdown(id="demo-values", 
                                options=[{"label":i, "value":i} for i in x_test[value].unique()],
                                value = "",
                                clearable=False)
        else:
            return dcc.Input(id="demo-values",
                            type="text",
                            value="",
                            debounce=True)

    @app.callback(
        [Output("cohort-metrics-div", "children"),
        Output("cohort-details", "children"),
        Output("cohort-graph", "children")],
        [Input("add-cohort","n_clicks"),
        Input("remove-cohort","n_clicks"),
        Input("x-axis","value")],
        [State("demo-dropdown","value"),
        State("demo-operators", "value"),
        State("demo-values", "value")],
    )
    def cohort_metrics_details(add_cohort, remove_cohort, x_axis, var_name, operator, value):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'remove-cohort' in changed_id:
            ca.remove_cohort()
            fig = ca.cohort_graph(x_axis)
            return ca.cohort_metrics_details(), ca.cohort_details(), dcc.Graph(figure=fig), 

        else:
            ca.add_cohort_metrics(x_test, var_name, operator,value)
            cohort = ca.add_cohort(x_test, x_axis, var_name, operator, value)
            fig = ca.cohort_graph(x_axis)
            return ca.cohort_metrics_details(), ca.cohort_details(), dcc.Graph(figure=fig)
    
    return layout 
