from imports import *
from rescale_numeric_feature import *
"""
This class helps to plot summary plot

Input: 


"""


class summary_plot():
    def __init__(self):
        super(summary_plot, self).__init__()
        self.param= None
        self.original_columns= None


    def find(self,  df):

        column = get_cols()
        self.original_columns = column.get_all_cols(df)

        re= rescale_numeric_features()
        df_with_rescaled_features= re.rescale(df)

        final_dataframe= self.rearrange_dataframe(df_with_rescaled_features )

        return final_dataframe

    def rearrange_dataframe(self, df_re ):

        df_final = pd.DataFrame()

        for v in self.original_columns:
            df_single = df_re[[v, v + '_rescaled', v + '_impact']]
            df_single["Variable Name"] = v
            df_single.columns = ['Original Feature Value', 'Rescaled Feature Value', 'Feature Impact on Outcome', 'Feature Name']
            df_final = pd.concat([df_final, df_single])

        return df_final





