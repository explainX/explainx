from imports import *
from plotly_graphs import *
from plotly_css import *
from app import app

def cohort_layout(original_variables):

    var_name_dropdown = html.Div([
    html.P("Choose Variable"),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[{'label': i, 'value': i} for i in original_variables],
        value= "",
        clearable=False
    )
])

    operators_list = ["==",">","<",">=","<="]

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
        dcc.Input(id="demo-values",
                type="text",
                value="",
                debounce=True)
    ])

    x_axis_dropdown = html.Div([
        html.P("Choose X-Axis Variable"),
        dcc.Dropdown(id="x-axis",
                    options = [{"label":i, "value":i} for i in original_variables[-4:]],
                    value = original_variables[-2],
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

    
    layout = html.Div(
        [
            heading,
            details,
            x_axis_dropdown,
            html.Div([button, remove_button],style={"padding":"20px", "display":"flex"}),
            cohort_details,
            cohort_metrics_div,
            modal,
            html.Div(id="cohort-graph")
            
        ], id="main"
    )

    return layout