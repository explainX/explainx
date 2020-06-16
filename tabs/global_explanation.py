import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd

gloabal_explanation = html.Div([
    html.Div([
         html.H4('Data', 
                style={'backgroundColor': '#fff', 
                       'color':'black',
                       'border-radius': '15px 15px 0px 0px ',
                       'textAlign':'left',
                       'height':"50px",
                       'margin':'auto',
                       'padding-left': '70px',
                       'padding-top':'20px',
                       "font-family": "Helvetica, Arial, sans-serif",

                        }),
        html.Div([
            dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
            editable=True,
            filter_action="native",
            #sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 10,
            style_table={'overflowX': 'auto', 'margin': 'auto',"padding-left":'0px','width':'90%'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'fontWeight': 'normal',
                "fontSize" : "15px",
                'marginLeft':"10px",
                'color':'white'
            },
            style_cell={
#                 "font-family" : "Arial, Helvetica, sans-serif",
                "font-family": "Helvetica, Arial, sans-serif",
                "fontSize" : "11px",
                'width': '{}%'.format(len(df.columns)),
                'textOverflow': 'ellipsis',
                'overflow': 'hidden',
                'textAlign':'left',
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
            }],

            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in df.to_dict('rows')
            ],
            tooltip_duration=None,

        )
        ])

        ], style={'backgroundColor':"#fff", "minHeight":"500px",'border-radius': '15px 15px 15px 15px',
                 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                 'border-right':'1px solid #2c3e50', 'border-bottom':'1px solid #2c3e50', 'marginTop':50}),
    
    html.Div(id='datatable-interactivity-container', style={'height':'400'})
        
    ])

        