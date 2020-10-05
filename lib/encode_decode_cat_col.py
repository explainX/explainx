from imports import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np
"""
This class calculates feature importance

Input: 


"""


class encode_decode_cat_col():
    def __init__(self):
        super(encode_decode_cat_col, self).__init__()
        self.param = None
        self.column_names = {}
        self.column_names_array = {}
        self.one_hot_encoder = {}

    def encode_col(self, df, column_name):
        if column_name in self.column_names.values():
            df = self.encode_existing(df, column_name)
        else:
            df = self.encode_new(df, column_name)

        return df

    def encode_new(self, df, column_name):
        self.one_hot_encoder[column_name] = OneHotEncoder(handle_unknown='ignore')

        # fill missing values with the a new category missing
        df[column_name] = df[column_name].fillna("missing")

        en = self.one_hot_encoder[column_name].fit_transform(df[[column_name]]).toarray()
        df_en = pd.DataFrame(en)

        # change column_names
        self.column_names[column_name], self.column_names_array[column_name] = self.change_column_names(df_en,
                                                                                                        column_name)
        df_en = df_en.rename(columns=self.column_names[column_name])

        # add new column and merge enc with the existing dataframe
        final_df = df.join(df_en)

        final_df = final_df.drop(column_name, axis=1)

        return final_df

    def encode_existing(self, df, column_name):

        # fill missing values with the a new category missing
        df[column_name] = df[column_name].fillna("missing")

        en = self.one_hot_encoder[column_name].fit_transform(df[[column_name]]).toarray()
        df_en = pd.DataFrame(en)

        # change column_names
        df_en = df_en.rename(columns=self.column_names[column_name])

        # add new column and merge enc with the existing dataframe
        final_df = df.join(df_en)

        final_df = final_df.drop(column_name, axis=1)

        return final_df

    def decode_col(self, df, column_name):

        # get columns
        df_en = df[self.column_names_array[column_name]]

        # get model
        arr = df_en.to_numpy()
        re_en = self.one_hot_encoder[column_name].inverse_transform(arr)

        df2 = pd.DataFrame(re_en)

        df2 = df2.rename(columns={0: column_name})

        # merge new col to the df
        final_df = df.join(df2)

        # remove existing columns
        final_df = final_df.drop(self.column_names_array[column_name], axis=1)

        return final_df

    def change_column_names(self, df, column_name):
        new_columns_names = {}
        col_array = []
        for i in list(df.columns):
            new_columns_names[i] = column_name + "_" + str(i)
            col_array.append(column_name + "_" + str(i))
        return new_columns_names, col_array





