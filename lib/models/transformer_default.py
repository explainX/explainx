import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Union
from sklearn.impute import SimpleImputer

class defaultTransformer():
    def __init__(self, data):
        self.data = data  
    
    def num_cat_variables(self):
        """ [Get categorical & numerical columns]
        
        Args:
            x_test
        
        Return:
            category_columns
            numerical_columns
        """
        is_cat = np.array([dt.kind == "O" for dt in self.data.dtypes])
        cat_cols = self.data.columns.values[is_cat]
        num_cols = self.data.columns.values[~is_cat]
        return cat_cols, num_cols
    
    def gen_category_map(self, input_data: Union[pd.DataFrame, np.ndarray],
                         categorical_columns: Union[List[int], List[str], None] = None) -> Dict[int, list]:
        """
        Parameters
        ----------
        data
            2-dimensional pandas dataframe or numpy array.
        categorical_columns
            A list of columns indicating categorical variables. Optional if passing a pandas dataframe as inference will
            be used based on dtype 'O'. If passing a numpy array this is compulsory.
        Returns
        -------
        category_map
            A dictionary with keys being the indices of the categorical columns and values being lists of categories for
            that column. Implicitly each category is mapped to the index of its position in the list.
        """
       
        if input_data.ndim != 2:
            raise TypeError('Expected a 2-dimensional dataframe or array')
        n_features = input_data.shape[1]

        if isinstance(input_data, np.ndarray):
            # if numpy array, we need categorical_columns, otherwise impossible to infer
            if categorical_columns is None:
                raise ValueError('If passing a numpy array, `categorical_columns` is required')
            elif not all(isinstance(ix, int) for ix in categorical_columns):
                raise ValueError('If passing a numpy array, `categorical_columns` must be a list of integers')
            input_data = pd.DataFrame(input_data)

        # infer categorical columns
        if categorical_columns is None:
            try:
                categorical_columns = [i for i in range(n_features) if input_data.iloc[:, i].dtype == 'O']  # NB: 'O'
            except AttributeError:
                raise

        # create the map
        category_map = {}
        for col in categorical_columns:
            if not isinstance(col, int):
                col = int(input_data.columns.get_loc(col))
            le = LabelEncoder()
            try:
                _ = le.fit_transform(input_data.iloc[:, col])
            except (AttributeError, IndexError):
                raise

            category_map[col] = list(le.classes_)

        return category_map
    
    def column_processor(self):
        '''
        Input:
        data : dataframe or x_test
        Column Transformer: Sklearn ColumnTransformer that applies transformers to columns of an array or pandas DataFrame.

        Returns:
        Column Transformer function after fit. 
        '''
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.compose import ColumnTransformer
        
        category_cols, numerical_cols = self.num_cat_variables()
        category_map = self.gen_category_map(self.data, category_cols)

        
        ordinal_features = [x for x in range(len(self.data.columns)) if x not in list(category_map.keys())]
        ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
        
        categorical_features = list(category_map.keys())
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(drop='first', handle_unknown='error'))])
        
        #pass the pipeline into the transformer
        preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                            ('cat', categorical_transformer, categorical_features)])
        #preprocessor.fit(data)
        return preprocessor