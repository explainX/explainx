from imports import *
from encode_decode_cat_col import *
from rescale_numeric_feature import *
from PDASH import ProtodashExplainer
from PDASH_utils import *
"""
This class calculates similar prototypes

Input: 


"""


class protodash():
    def __init__(self):
        super(protodash, self).__init__()
        self.encode= None
        self.get_col=None
        self.cat_col=[]
        self.actual_variables= []
        self.classification= False
        self.df=None
        self.y_variable= None



    def encode_categorical_var(self, df):
        if self.get_col==None:
            self.get_col= get_cols()
            num_col, self.cat_col = self.get_col.get_cate_numer_col(df)

        if self.encode==None:
            self.encode= encode_decode_cat_col()

        for col_name in self.cat_col:
            df = self.encode.encode_col(df, col_name)

        return df

    def decode_categorical_var(self, df):
        # let's decode
        for col_name in self.cat_col:
            df = self.encode.decode_col(df, col_name)

        return df



    def preprocess_data(self,  df, y_variable):

        #store y_var
        self.y_variable=y_variable

        # encode categorical var
        df= self.encode_categorical_var(df)
        
        # check and remove index column in the file.
        try:
            df = df.drop("index", axis=1)
        except:
            pass

        # finad and remove variables containing shap values
        self.df= self.remove_impact_columns(df)

        self.classification= self.is_classification()

        return True




    def remove_impact_columns(self, df):
        self.actual_variables = [col for col in df.columns if not '_impact' in col]
        self.actual_variables = [col for col in self.actual_variables if not '_rescaled' in col]
        df = df[self.actual_variables]
        return df


    def is_classification(self):
        count = self.df[self.y_variable].nunique()

        if count>=10:
            return False
        else:
            return True

    def find_prototypes(self, row_number):

        # get training data
        data, Z= self.z_train_good(row_number)
        #get prototypes
        explainer = ProtodashExplainer()
        try:
            (W, S, setValues) = explainer.explain(Z, data, m=5, kernelType='other', sigma=5)
        except:
            (W, S, setValues) = explainer.explain(Z, data, m=5, kernelType='other', sigma=5) #Guassian gives an error. 
        #make a dataframe
        dfs = pd.DataFrame.from_records(data[S, 0:].astype('double'))
        dfs.columns = self.actual_variables
        dfs["Weight(%)"] = ((np.around(W, 5) / np.sum(np.around(W, 5)))*100)
        dfs = self.decode_categorical_var(dfs)
        return dfs,self.df.iloc[row_number]

    def z_train_good(self, row_number):
        row= self.df.iloc[row_number]
        #remove row from the data
        df = self.df
        # df = self.df.drop(self.df.index[row_number])
        if self.classification==True:
            predict_value= row[self.y_variable]
            df= df[df[self.y_variable] == predict_value]  # choose prediction: 0/1
        row1 = row.values
        array= df.values
        Z= row1.reshape(-1, 1).T

        return array, Z



