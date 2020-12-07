from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Union
from sklearn.impute import SimpleImputer
from explainx.lib.utils import is_classification

class ModelProcessor():
    def __init__(self, model, x_test, y_test, ct):
        
        super(ModelProcessor, self).__init__()
        
        self.predicted_columns = {}
        self.model = model
        self.input_data = x_test
        self.target_data = y_test
        self.ct = ct
          
    def num_cat_variables(self, data):
        """ [Get categorical & numerical columns]
        
        Args:
            x_test
        
        Return:
            category_columns
            numerical_columns
        """
        is_cat = np.array([dt.kind == "O" for dt in data.dtypes])
        cat_cols = data.columns.values[is_cat]
        num_cols = data.columns.values[~is_cat]
        return cat_cols, num_cols
    
      

    def gen_category_map(input_data: Union[pd.DataFrame, np.ndarray],
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
    
    def columnsTransformer(self):
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
        
        category_cols, numerical_cols = self.num_cat_variables(self.input_data)
        category_map = self.gen_category_map(self.input_data, category_cols)

        if self.ct == None:
            pass
        elif self.ct == "default":
            ordinal_features = [x for x in range(len(self.input_data.columns)) if x not in list(category_map.keys())]
            ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])
            
            categorical_features = list(category_map.keys())
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(drop='first', handle_unknown='error'))])
            
            #pass the pipeline into the transformer
            preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                               ('cat', categorical_transformer, categorical_features)])
            preprocessor.fit(self.input_data)
            return preprocessor
        else:
            transformerFunction = ct.fit(self.input_data)
            #transformed_xtest = ct.transform(x_test)
            return transformerFunction
    
    def inbuiltColumnsTransformer(data):
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
        
        category_cols, numerical_cols = ModelProcessor.num_cat_variables(data)
        category_map = ModelProcessor.gen_category_map(data, category_cols)

        
        ordinal_features = [x for x in range(len(data.columns)) if x not in list(category_map.keys())]
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
        
    
    def data_into_model(self):
        ''' [Transformed data that is directly passed into the model]
        Args:
        data : dataframe or x_test
        Column Transformer: Sklearn ColumnTransformer that applies transformers to columns of an array or pandas DataFrame.

        Returns:
        x_test_proc: transformed dataset to be passed into the model predict function
        '''
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.compose import ColumnTransformer

        if self.ct == None:
            pass
        
        elif self.ct=="default":
            x_test_proc = self.columnsTransformer().transform()
            return x_test_proc
            
        else:
            #transformerFunction = ct.fit(x_test)
            x_test_proc = self.ct.transform(self.input_data)
            return x_test_proc
    
    def make_predictions(self):
        """ [Initiate the prediction function]
        
        Args:
            Model
            input_data
            target_data
        
        Return:
            prediction column
            probabilities [if is_classifier]
        
        """
        if is_classification(self.model):
            if self.ct == None:
                prediction = self.model.predict(self.input_data.to_numpy())
                probabilities = self.model.predict_proba(self.input_data.to_numpy())
                return prediction, probabilities
            elif self.ct != None: 
                prediction = self.model.predict(self.data_into_model())
                probabilities = self.model.predict_proba(self.data_into_model())
                return prediction, probabilities
            else:
                raise Exception(("{} not supported. Please create an issue on Github").format(self.model))
            
        else:
            if self.ct == None:
                prediction = self.model.predict(self.input_data)
                return prediction
            elif self.ct != None: 
                prediction = self.model.predict(self.data_into_model())
                return prediction
            else:
                raise Exception(("{} not supported. Please create an issue on Github").format(self.self.model))
            
    
    def create_prediction_columns(self):
        """ [Create prediction columns and add them to the self.predicted_columns dictionary]
        Args:
            model
            x_test: 
            y_test
            ColumnTransformer
        """
        if is_classification(self.model) == True:
            prediction, probabilities = self.make_predictions()
            self.predicted_columns['Model Decision'] = prediction
            self.predicted_columns['True Values'] = self.target_data
            for i in range(len(np.unique(prediction))):
                self.predicted_columns['Probability: {}'.format(np.unique(prediction)[i])] = probabilities[:,i]
            
        else:
            prediction = self.make_predictions()
            self.predicted_columns['Model Decision'] = prediction
            self.predicted_columns['True Values'] = self.target_data
            
            
    def log_metrics(self):
        if is_classification(self.model) == True:
            predict, _ = self.make_predictions()
            metrics = self.classification_metrics(self.target_data, predict)
            return metrics
        else:
            predict = self.make_predictions()
            metrics = self.regression_metrics(self.target_data, predict)
            return metrics
            
    def classification_metrics(self, target_data, predicted): 
        """[Calculates the metrics for classification problems]

        Args:
            y_true ([type]): [True labels from the dataset]
            predicted ([type]): [Predicted values from the model]

        Returns:
            Accuracy metric of the model
            Precision value of the model
            Recall value of the model
            False Positive Rate
            False Negative Rate
        """
        from sklearn import preprocessing
        from sklearn import metrics 

        y_true_copy, predictions = pd.DataFrame(self.target_data), predicted
        #y_true_copy.unique()
        np.unique(y_true_copy)
        encode = {}
        for i in range(len(np.unique(y_true_copy))):
            encode[np.unique(y_true_copy)[i]] = i
        
        predicted_copy = [encode[i] for i in predictions]
        
        y_true_copy.replace(encode, inplace=True)
        
        if len(y_true_copy) != 0:
            #Accuracy
            accuracy = round(metrics.accuracy_score(y_true_copy, predicted_copy),2) 
            #Precision
            precision = round(metrics.precision_score(y_true_copy, predicted_copy, zero_division=1),2) 
            #Recall
            recall = round(metrics.recall_score(y_true_copy, predicted_copy, zero_division=1),2)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true_copy, predicted_copy).ravel()
            #False Positive Rate (FPR)
            fpr = round((fp/(fp+tn)),2)
            #Flase Negative Rate (FNR)
            fnr = round((fn/(tp+fn) if (tp+fn) else 0),2) 
            results = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'fpr': fpr, 'fnr':fnr}
            return results
        else:
            raise Exception("Metrics calculation failed")
    
    def regression_metrics(self, target_data, predicted):
        """[Calculates the metrics for regression problems]

        Args:
            y_true ([type]): [True labels from the dataset]
            predicted ([type]): [Predicted values from the model]

        Returns:
            Mean Absolute Error
            Mean Squared Error
            R-Squared Value
        """
        from sklearn import metrics 
        if len(target_data) != 0:
            #Mean Absolute Error
            mae = round(metrics.mean_absolute_error(target_data, predicted),2)
            #Mean Squared Error
            mse = round(metrics.mean_squared_error(target_data, predicted),2)
            #R2
            r2 = round(metrics.r2_score(target_data, predicted),2)
            results = {'mae':mae, 'mse':mse, 'R2':r2}
            return results
        else:
            raise Exception("Metrics calculation failed")

                
# if __name__ == '__main__':
#     a_game = ModelProcessor()
#     a_game.run()