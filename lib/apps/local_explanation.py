from imports import *
from plotly_graphs import *
from plotly_css import *
from protodash import *
from app import app
import pandas as pd

def layout_local(original_variables,columns,df_column):
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
                                        'width': '{}%'.format(len(df_column)),
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
                                    
                                )
                            )

                        ])

                    ], style=style39),

                ])

                    ])

                ], style={"width":'69%'})

            ], style={'display':'flex'}),
        
             ])
    

    return layout




