from explainx import *
from explainx.lib.imports import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json
from explainx.lib.models.transformer_default import defaultTransformer
from explainx.lib.models.modelprocessor import ModelProcessor
from explainx.lib.modules.local_explanation.index import localExplanation
from explainx.lib.modules.cohort_analysis.index import cohort
from explainx.lib.modules.feature_interactions.index import featureInteraction
from explainx.lib.frameworks.shapley_values import ShapleyValues


class ExplainxModules:
    def __init__(self):
        self.model = None
        self.input_data = None
        self.target_data = None
        self.ct = None
        self.ModelProcessor = None
        self.ShapleyValues = None

    def ai(self, model, x_test, y_test, ct=None):
        self.model = model
        self.input_data = x_test
        self.target_data = y_test
        self.ct = ct

        self.ModelProcessor = ModelProcessor(self.model, self.input_data, self.target_data, self.ct)
        self.ShapleyValues = ShapleyValues(self.model, self.input_data, self.target_data, self.ct)

    def default_transformer(self):
        return self.ModelProcessor.columnTransformer(data)

    def predicted_columns(self):
        '''
        return true values in a dict format
        '''
        self.ModelProcessor.create_prediction_columns()
        return self.ModelProcessor.predicted_columns

    def dataframe_graphing(self):
        '''
        return dataframe for graphing
        '''
        main_dataset = self.input_data.copy()
        for i in self.predicted_columns().keys():
            main_dataset[i] = self.predicted_columns()[i]
        return main_dataset

    def metrics(self):
        '''
        returns metrics as a dictionary
        '''
        return self.ModelProcessor.log_metrics()

    def shap_df(self):
        '''
        global level explanation
        '''
        _, _, df_with_shap = self.ShapleyValues.global_shap_plotting()
        return df_with_shap

    def what_if_analysis(self, mode=None):
        return localExplanation(self.input_data, self.ShapleyValues, self.shap_df()).main_function(mode)

    def feature_interactions(self, mode=None):
        return featureInteraction(self.dataframe_graphing(), self.shap_df()).main_function(mode)

    def cohort_analysis(self, mode=None):
        return cohort(self.dataframe_graphing(), self.model).main_function(mode)


explainx_modules = ExplainxModules()
