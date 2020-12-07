from explainx.lib.imports import *
from explainx.lib.plotly_graphs import *
from explainx.lib.plotly_css import *
import dash_core_components as dcc
import dash_html_components as html

def layout_interaction(x_test, df3, app):

    layout = html.Div([

        html.Div([
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
                                        options = [{'label': i, 'value': i} for i in x_test.columns],
                                        value = x_test.columns[1],
                                        clearable=False
                                    ),

                                ],style=style22
                                    # style={'width': '20%', 'marginLeft': 70, 'float': 'left',
                                    #         'display': 'inline-block'}
                                            ),

                                html.Div([
                                    html.P('Color Axis'),
                                    dcc.Dropdown(
                                        id='third-axis',
                                        options=[{'label': i, 'value': i} for i in x_test.columns],
                                        value=x_test.columns[-3],
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
                                        options=[{'label': i, 'value': i} for i in x_test.columns[1]],
                                        value=x_test.columns[1],
                                        clearable=False
                                    )
                            ], style={'display':'none'}),
                                html.P(
                                    'In the summary plot, we see first indications of the relationship between the value of a feature and the impact on the prediction',
                                    style=style18)
                                ,
                             
                                    dcc.Graph(id='summary_plot', style={'marginLeft': 50, 'height': '600px'})

                                # ),

                            ], style=style19),
                        ], style=style20)
                    )
                ])

    ])
    

    # Partial Dependence Plot Graph
    @app.callback(
        Output('indicator-graphic', 'figure'),
        [Input('xaxis-column', 'value'),
            Input('third-axis', 'value')])
    def update_graph(xaxis_column_name, third_axis_name):
       
        g = plotly_graphs()
        fig = g.pdp_plot(df3, df3[xaxis_column_name], df3[xaxis_column_name+"_impact"], df3[third_axis_name])
        return fig

    # Summary Plot
    @app.callback(
    Output('summary_plot', 'figure'),
    [Input('xaxis-column', 'value')])
    def update_graph2(value):

        g = plotly_graphs()
        graph_type = 'summary_plot'
        #df3 = self.caching_data_manager(df3, sql_query, graph_type, g.summary_plot)
        df4 = g.summary_plot(df3)
        fig = g.summary_plot_graph(df4)
        return fig

    return layout
