from imports import *
from plotly_graphs import *
from plotly_css import *
from app import app

def layout_interaction(original_variables, y_variables):
    layout =  html.Div([
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

                                ],style=style22
                                    # style={'width': '20%', 'marginLeft': 70, 'float': 'left',
                                    #         'display': 'inline-block'}
                                            ),

                                # html.Div([
                                #     html.P("Variable Impact Values"),
                                #     dcc.Dropdown(
                                #         id='yaxis-column',
                                #         options=[{'label': i, 'value': i} for i in y_variables],
                                #         value=y_variables[1],
                                #         clearable=False
                                #     ),

                                # ], style=style14),

                                html.Div([
                                    html.P('Color Axis'),
                                    dcc.Dropdown(
                                        id='third-axis',
                                        options=[{'label': i, 'value': i} for i in original_variables],
                                        value=original_variables[-3],
                                        clearable=False
                                    ),

                                ], style=style23),

                            ]),
                            dcc.Loading(
                                id="loading-5",
                                type="circle",
                                children=dcc.Graph(id='indicator-graphic', style={'marginLeft': 50, 'marginTop':80})
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
                                html.Div([
                                 dcc.Dropdown(
                                        id='xaxis-column-test',
                                        options=[{'label': i, 'value': i} for i in original_variables[1]],
                                        value=original_variables[1],
                                        clearable=False
                                    )
                            ], style={'display':'none'}),
                                html.P(
                                    'In the summary plot, we see first indications of the relationship between the value of a feature and the impact on the prediction',
                                    style=style18)
                                ,
                                # dcc.Loading(
                                #     id="loading-6",
                                #     type="circle",
                                #     children=
                                    dcc.Graph(id='summary_plot', style={'marginLeft': 50, 'height': '600px'})

                                # ),

                            ], style=style19),
                        ], style=style20)
                    )
                ]),
    return layout
