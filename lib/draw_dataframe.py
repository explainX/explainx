from imports import *


class draw_dataframe():
    def __init__(self):
        super(draw_dataframe, self).__init__()
        self.param= None


    def draw(self,  dataframe, max_rows=10):
        return self.generate_table(dataframe, max_rows)


    # Generating table from pandas dataframe
    def generate_table(self, dataframe, max_rows):
        return html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ])

