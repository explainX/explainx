import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd


feature_interaction_layout = html.Div([
                        html.Div([
                            html.H4('Partial Dependence Plot', 
                                style={'backgroundColor': '#fff', 
                                       'color':'black',
                                       'border-radius': '15px 15px 15px 15px ',
                                       'textAlign':'left',
                                       'height':"50px",
                                       'margin':'auto',
                                        'padding-left': '70px',
                                        'padding-top':'20px'

                                        }),
                            html.Div([
                                html.P('Variable Name'),
                                dcc.Dropdown(
                                    id='xaxis-column',
                                    options=[{'label': i, 'value': i} for i in available_columns],
                                    value='Cabin'
                                ),

                            ],
                            style={'width': '25%', 'marginLeft':70,'float': 'left' ,'display': 'inline-block'}),

                            html.Div([
                                html.P("Variable Impact Values"),
                                dcc.Dropdown(
                                    id='yaxis-column',
                                    options=[{'label': i, 'value': i} for i in y_variables],
                                    value='Cabin_impact'
                                ),

                            ],style={'width': '25%',  'float': 'center', 'display': 'inline-block'}),

                            html.Div([
                                html.P('3rd Variable'),
                                dcc.Dropdown(
                                    id='third-axis',
                                    options=[{'label': i, 'value': i} for i in available_columns],
                                    value='predict'
                                ),

                            ],style={'width': '25%', 'marginRight':10,'float': 'center', 'display': 'inline-block'})
                        ]),

                    dcc.Graph(id='indicator-graphic', style={'marginLeft':50}), 
                    ],
                         style={'backgroundColor':"#fff", "minHeight":"600px",'border-radius': '15px 15px 15px 15px',
                                 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                 'border-right':'1px solid #2c3e50', 'border-bottom':'1px solid #2c3e50', 'marginTop':50}),

                        html.Div([
                            html.Div([
                                html.H4('Summary Plot', 
                                style={'backgroundColor': '#fff', 
                                       'color':'black',
                                       'border-radius': '15px 15px 15px 15px ',
                                       'textAlign':'left',
                                       'height':"50px",
                                       'margin':'auto',
                                        'padding-left': '70px',
                                        'padding-top':'20px'

                                        })
                                ,
                                dcc.Graph(id="a",style={'marginLeft':50},

                                figure=summary_plot.update_layout({
                                     "xaxis": {"automargin": True, "title":{"text":"Impact"}},
                                     "yaxis": {
                                            "automargin": True,
                                            "title": {"text": 'Variable Name'}
                                        },

                                        "margin": {"t": 10, "l": 100, "r": 100},
                                })   
                            ),

                            ], style={'backgroundColor':"#fff", "minHeight":"600px",'border-radius': '15px 15px 15px 15px',
                                 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                                 'border-right':'1px solid #2c3e50', 'border-bottom':'1px solid #2c3e50', 'marginTop':50,}),
                        ]) 
