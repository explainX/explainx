from imports import *

"""
This class calculates feature importance

Input: 


"""


class get_cols():
    def __init__(self):
        super(get_cols, self).__init__()
        self.param = None

    def get_all_cols(self, df):
        all_columns = df.columns

        original_columns = []
        for c in all_columns:
            if "_impact" in c:
                pass
            elif c + "_impact" in all_columns:
                original_columns.append(c)

        return original_columns

    def get_cate_numer_col(self, df):

        original_columns = self.get_all_cols(df)

        categorical_columns = self.cate_col(df)
        numeric_columns = set(original_columns) - set(categorical_columns)
        numeric_columns = list(numeric_columns)
        return numeric_columns, categorical_columns

    def cate_col(self, df):
        df= df.convert_dtypes()
        di = dict(df.dtypes)
        c = df.columns

        sample_df = pd.DataFrame(["a", "b", "c"])
        sample_df = sample_df.convert_dtypes()
        sample_df_dic= dict(sample_df.dtypes)

        cate = []
        for i in c:
            try:
                if di[i] == sample_df_dic[0]:
                    cate.append(i)
            except:
                pass

        return cate

    def cate_col_with_index(self, df):
        df = df.convert_dtypes()
        di = dict(df.dtypes)
        c = df.columns

        sample_df = pd.DataFrame(["a", "b", "c"])
        sample_df = sample_df.convert_dtypes()
        sample_df_dic = dict(sample_df.dtypes)

        cate = []
        index=[]
        j=0
        for i in c:
            try:
                if di[i] == sample_df_dic[0]:
                    cate.append(i)
                    index.append(j)
            except:
                pass
            j = j + 1


        return cate, index


class rescale_numeric_features():
    def __init__(self):
        super(rescale_numeric_features, self).__init__()
        self.param = None

    def get_min_max(self, df_describe, variable_name):
        mini = df_describe[variable_name][3]
        maxi = df_describe[variable_name][-1]
        return mini, maxi

    def add_col_rescaled(self, df):

        df_describe = df.describe()

        # let's find numeric and categorical columns
        column = get_cols()
        numeric_columns, categorical_columns = column.get_cate_numer_col(df)

        for nc in numeric_columns:
            # get min and max
            mini, maxi = self.get_min_max(df_describe, nc)

            df[nc + "_rescaled"] = (df[nc] - mini) / (maxi - mini) * 10

        for cc in categorical_columns:
            df[cc + "_rescaled"] = 0

        return df

    def rescale(self, df):
        column = get_cols()
        original_columns = column.get_all_cols(df)

        df_with_rescaled_features = self.add_col_rescaled(df)

        return df_with_rescaled_features






