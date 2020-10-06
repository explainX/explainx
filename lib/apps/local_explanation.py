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
                html.Div([
                    html.H4('What-If Analysis',
                            style=style26),
                    html.I("Datapoint Index."),

                    dcc.Input(id="row_number", type="number", value=1,
                                placeholder="Enter a Row Number e.g. 1, 4, 5",
                                style={'text-align': 'center'}),
                    html.Button(id='submit-button-state', n_clicks=0, children='Predict', style={'margin-left':'50px'}),
                    

                    # html.Div(id='local_data_table', style={'marginTop': "20px", 'marginBottom': "5px"}),


                ]),

            ]),  # End of Input & Data Div

            html.Div([

                #What-If Div
                html.Div([

                    

                    html.Br(),
                    html.Div([
                        
                        dbc.Table(html.Thead(
                            html.Tr(
                                [
                                    html.Th("Feature", style={"width":"50%", 'marginLeft':'10px'}), 
                                    html.Th("Value")])), 
                        bordered=True, 
                        dark=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                        style={'width':'90%'}),

                    html.Div(id="place_form_here", 
                        style={ 
                            "maxHeight": "740px", 
                             "maxWidth":"550px",
                            "overflow": "scroll",
                            # "border-right":"1px solid grey",
                            # "border-bottom":"1px solid grey",
                            # "border-top":"1px solid grey",
                            "border-radius":'0 0 10px 10px',
                            'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                            "display":"flex",
                            "justify-content":"center",
                            "padding-top":"10px",
                            "padding-bottom":"10px"}),
                   
                      
                      ])

                ], style={'width':'30%'}),

                #Tabs Div
                html.Div([

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
                                                                        style={
                                                                            # 'marginLeft': 0,
                                                                            #     'marginTop': 0,
                                                                                'height': '600px'})),
                                                    width=8),
                                            dbc.Col(
                                                [
                                                    # html.Div([
                                                    #     html.H4("How to read this graph?"),
                                                    #     html.P(
                                                    #         "According to the model, the features are most important in explaining the target variable. Most importance is on the top.")
                                                    # ]),
                                                    html.Div([
                                                        html.H4("Insights"),
                                                        html.P(id='local_message_1'),
                                                        html.P(id='local_message_2'),
                                                        html.P(id='local_message_3')
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
                                        # 'width': '{}%'.format(len(df.columns)),
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

                ], style={"width":'70%'})

            ], style={'display':'flex'}),
        
             ])
    

    return layout




