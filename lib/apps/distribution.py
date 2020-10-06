from imports import *
from plotly_graphs import *
from plotly_css import *
from app import app


def layout_distribution(original_variables):
    layout = html.Div([
            html.Div([
                html.Div([
                    html.H4('Distributions',
                            style=style21),
                    html.Div([
                        html.P('Variable Name'),
                        dcc.Dropdown(
                            id='xaxis-column-name',
                            options=[{'label': i, 'value': i} for i in original_variables],
                            value= original_variables[0]
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

            # html.Div([

            #     html.Div([
            #         html.H4('Multi-Level EDA', style=style21),
            #         html.Div([
            #             html.P('X-Axis Variable'),
            #             dcc.Dropdown(
            #                 id='x_axis',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 value=original_variables[0]
            #             ),
            #         ], style=style22),
            #         html.Div([
            #             html.P('Y-Axis Variable'),
            #             dcc.Dropdown(
            #                 id='y_axis',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 value=original_variables[1]
            #             ),
            #         ], style=style22),
            #         html.Div([
            #             html.P('Size'),
            #             dcc.Dropdown(
            #                 id='size',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 # value=original_variables[0]
            #                 placeholder="Choose a variable"
            #             ),
            #         ], style=style22),
            #         html.Div([
            #             html.P('Color'),
            #             dcc.Dropdown(
            #                 id='color',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 placeholder="Choose a variable"
            #             ),
            #         ], style=style22),
            #         html.Div([
            #             html.P('Drill-down by Column'),
            #             dcc.Dropdown(
            #                 id='facet_col',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 placeholder="Select a categorical variable"
            #                 # value=original_variables[0]
            #             ),
            #         ], style=style22),
            #         html.Div([
            #             html.P('Drill-down by Row'),
            #             dcc.Dropdown(
            #                 id='facet_row',
            #                 options=[{'label': i, 'value': i} for i in original_variables],
            #                 placeholder="Select a categorical variable"
            #             ),
            #         ], style=style22)
            #     ]),
            #     dcc.Graph(id='multi_level_eda',
            #                 style={'marginLeft': 50, 'marginTop': 200, 'height': '700px'})

            # ], style=style24)
        ], style=style25)
    return layout
