from explainx.lib.imports import *
from explainx.lib.plotly_css import *
from explainx.lib.plotly_graphs import *
from explainx.lib.protodash import *
from explainx.lib.frameworks.shapley_values import ShapleyValues
from explainx.lib.utils import is_classification
from .insights import *
from .insight_classification import *
from .insight_regression import *
from .what_if import *

def layout_local(ShapleyValues, x_test, df_with_shap, app):

    layout = html.Div([
            # Input and Data
                html.Div([
                        html.H3('What-If Analysis'),
                        html.Div([
                            html.P("Datapoint Index: "),
                            dcc.Input(
                                id="row_number", 
                                type="number", 
                                value=1,
                                placeholder="Enter a Row Number e.g. 1, 4, 5",
                                style={'text-align': 'center'}),
                            html.Button(
                                id='submit-button-state', 
                                n_clicks=0, 
                                children='Predict', 
                                style={'margin-left':'10px'}),
                                ], style={"display":"flex"})
                        ],style={'margin-bottom':"10px"}),

            # End of Input & Data Div
            html.Div([
                #What-If Div
                html.Div([
                    html.Div([
                            dbc.Table(html.Thead(html.Tr([html.Th("Feature"), html.Th("Value")])), 
                            bordered=True, 
                            dark=True,
                            hover=True,
                            responsive=True,
                            striped=True,
                            style={'width':'100%'}),

                            html.Div(id="place_form_here", className="place_form")
                            ])
                ], style={'width':'29%'}),
            
                html.Div([], style={'width':"1%"}),

                #Tabs Div
                html.Div([dcc.Tabs([
                    dcc.Tab(label='Local Feature Explanation', children=[
                    html.Div(id='datatable-interactivity-container2', children=[
                        html.Div([
                            html.Div([
                                html.Div([
                                html.H4('Features Influencing This Predictions', className="local_impact_heading"),
                                html.P(
                                    'This graph identifies which features (also known as columns or inputs) in your dataset had a positive or negative influence on the final outcome of your machine learning model.', className="local_impact_details"),
                            ]),

                            
                                dcc.Loading(
                                    id="local_feature_impact_1",
                                    type="circle",
                                    children=dbc.Row(
                                        [
                                            dbc.Col(html.Div(dcc.Graph(id='local_feature_impact',
                                                                        style={'marginLeft': 10, 'height': '590px'})), width=8),
                                            dbc.Col(
                                                [
                                                    html.Div([
                                                        html.H4(id='local_message_1'),
                                                        html.H4(id='local_message_2'),
                                                        html.H5("How this prediction was determined?"),
                                                        html.H5(id='local_message_3'),
                                                        html.H5(id='local_message_4')
                                                    ]),
                                                   
                                                ]
                                                , width=4),

                                        ]))

                            ],
                                style=style31,
                            ),
                        ],style=style32)], style={'height': '400'})]),

                    ])

                ], style={"width":'69%'})

            ], style={'display':'flex'}),
        
             ])

    @app.callback(
            Output('place_form_here', 'children'),
            [Input('row_number', 'value')])
    def create_what_if_form(row_number):
        x = what_if()
        i = 0

        if type(row_number) == type(1):
            i = row_number

        array = df_with_shap[i:i+1]
        
        impact_variables = [col for col in array if '_impact' in col]
        for col in impact_variables:
            array.drop([col], axis=1, inplace=True)

        form = x.what_if_form(array, x_test.columns)
        return form
    
    
    callback_input = [Input(f + '_slider', 'value') for f in x_test.columns]
    callback_input.append(Input('submit-button-state', 'n_clicks'))
    callback_input.append(Input('row_number', 'value'))

    # Local Feature Impact Graph
    @app.callback(
        [Output('local_feature_impact', "figure"),
            Output('local_message_1', "children"),
            Output('local_message_2', "children"),
            Output('local_message_3', "children"),
            Output('local_message_4', 'children')],
            
        callback_input, prevent_initial_call=True)
    def update_impact_graph(*values):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        
        
        df = pd.DataFrame([values[:-2]])
        df.columns = x_test.columns

        array = ShapleyValues.add_shap_row(df, values[-1])
        g = plotly_graphs()
        figure, dat = g.local_feature_impact_graph(array)
        
        if is_classification(ShapleyValues.model):
            y_and_prob = []
            y_and_prob.append(int(array["Model Decision"]))
            y_and_prob.append(round(float(array["Probability: "+str(int(array["Model Decision"])) ]),2))
            
            message = insight_classification.insight_2_local_feature_impact(dat, y_and_prob)
        else:
            y_and_prob = []
            y_and_prob.append(int(array["Model Decision"]))
            message = insight_regression.insight_2_local_feature_impact(dat, y_and_prob)

        
        return figure, message[0], message[1], message[2], message[3]


    return layout




