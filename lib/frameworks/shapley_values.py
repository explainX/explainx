from operator import is_
import shap
from explainx.lib.utils import is_classification
import pandas as pd
import numpy as np
class ShapleyValues():
    
    def __init__(self, model,input_data, target_data, ct):
        #super().__init__(model, input_data, ct)
        
        self.model = model
        self.input_data = input_data
        self.actual_data = target_data
        self.ct = ct
        #self.row_number = 0
    
    def tree_explainer(self):
        explainer = shap.TreeExplainer(self.model)
        try:
            #classification case
            predictions = self.model.predict
            probabilities = self.model.predict_proba
            return explainer, predictions, probabilities
        except:
            #regression case
            predictions = self.model.predict
            return explainer, predictions
    
    def kernel_explainer(self):
        try:
            #classification case
            explainer = shap.KernelExplainer(self.model.predict_proba, 
                                                             shap.sample(self.input_data, 100),
                                                             link='logit',
                                                             feature_names=self.input_data.columns,
                                                            seed=0)
            predictions = self.model.predict
            prediction_probabilities = self.model.predict_proba
            return explainer, predictions, prediction_probabilities
        except:
            #regression case
            explainer = shap.KernelExplainer(self.model.predict, 
                                                             shap.sample(self.input_data, 100), 
                                                             link='identity',
                                                             feature_names=self.input_data.columns,
                                                            seed=0)
            predictions = self.model.predict
            return explainer, predictions
    
    def kernel_explainer_with_ct(self):
        try:
            #classification case
            pred_fcn = lambda x : self.model.predict_proba(self.ct.transform(x))
            explainer = shap.KernelExplainer(pred_fcn, shap.sample(self.input_data, 100), 
                                                                 link='logit',
                                                                 feature_names=self.input_data.columns,
                                                                 seed=0)
            pred = lambda x : self.model.predict(self.ct.transform(x))
            return explainer, pred, pred_fcn
        
        except:
            pred_fcn = lambda x : self.model.predict(self.ct.transform(x))
            explainer = shap.KernelExplainer(pred_fcn, shap.sample(self.input_data, 100), 
                                                                 link='identity',
                                                                 feature_names=self.input_data.columns,
                                                                 seed=0)
            return explainer, pred_fcn
    
    def shap_explainer(self):
        if is_classification(self.model):
            if self.ct == None:
                try:
                    explainer, pred, pred_prob = self.tree_explainer()
                    return explainer, pred, pred_prob
                except:
                    try:
                        explainer, pred, pred_prob = self.kernel_explainer()
                        return explainer, pred, pred_prob
                    except:
                        raise Exception(("{} not supported. Please create an issue on Github").format(self.model))
            else:
                try:
                    explainer, pred, pred_fcn = self.kernel_explainer_with_ct()
                    return explainer, pred, pred_fcn
                except:
                    raise Exception(("{} not supported. Please create an issue on Github").format(self.model))
        else:
            if self.ct == None:
                try:
                    explainer, pred = self.tree_explainer()
                    return explainer, pred
                except:
                    try:
                        explainer, pred = self.kernel_explainer()
                        return explainer, pred
                    except:
                        raise Exception(("{} not supported. Please create an issue on Github").format(self.model))
            else:
                try:
                    explainer, pred_fcn = self.kernel_explainer_with_ct()
                    return explainer, pred_fcn
                except:
                    raise Exception(("{} not supported. Please create an issue on Github").format(self.model))
            

    
    def append_shap_values_to_df(self, input_sv, in_data, scope):

        df_shap = pd.DataFrame(input_sv)
 
        features = list(self.input_data.columns)
        shap_columns = []
        for i in features:
            shap_columns.append(i + "_impact")

        try:
            df_shap.columns = shap_columns
        except: 
            df_shap = df_shap.T
            df_shap.columns = shap_columns

        input_data = in_data

        if scope == 'global':
            for i in shap_columns:
                input_data[i] = list(df_shap[i])
            return input_data
        else:
            for i in shap_columns:
                input_data[i] = list(df_shap[i])[0]
            return input_data
                
    def global_shap_plotting(self):
        if is_classification(self.model):
            explainer, pred, pred_fcn = self.shap_explainer()
            if type(explainer) == shap.explainers._tree.Tree:
                global_shap_values = explainer.shap_values(self.input_data)
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values[0], 
                                                                in_data=self.input_data.copy(), scope="global")
                prediction = pred(self.input_data)
                probabilities = pred_fcn(self.input_data)
                
                data_with_shap['Model Decision'] = prediction
                data_with_shap['True Values'] = self.actual_data
                for i in range(len(np.unique(prediction))):
                    data_with_shap['Probability: {}'.format(np.unique(prediction)[i])] = probabilities[:,i]
                return explainer, global_shap_values, data_with_shap
            else:
                predictions = pred(shap.sample(self.input_data,100))
                global_shap_values = explainer.shap_values(shap.sample(self.input_data,100))  
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values[0], 
                                                               in_data=shap.sample(self.input_data,100).copy(),
                                                              scope='global')  
                prediction = pred(shap.sample(self.input_data,100))
                probabilities = pred_fcn(shap.sample(self.input_data,100))
                data_with_shap['Model Decision'] = prediction
                data_with_shap['True Values'] = self.actual_data
                for i in range(len(np.unique(self.actual_data))):
                    data_with_shap['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i]

                return explainer, global_shap_values, data_with_shap
        else:
            explainer, pred = self.shap_explainer()
            if type(explainer) == shap.explainers._tree.Tree:
                #Complete! Do not change. 
                global_shap_values = explainer.shap_values(self.input_data)
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values, 
                                                                in_data=self.input_data.copy(),
                                                                  scope="global")
                data_with_shap['Model Decision'] = pred(self.input_data)
                data_with_shap['True Values'] = self.actual_data
                return explainer, global_shap_values, data_with_shap

            else:
                global_shap_values = explainer.shap_values(shap.sample(self.input_data,100))  
                predictions = pred(shap.sample(self.input_data,100))
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values, 
                                                               in_data=shap.sample(self.input_data,100).copy(),
                                                              scope="global")
                data_with_shap['Model Decision'] = pred(shap.sample(self.input_data,100))
                data_with_shap['True Values'] = shap.sample(self.actual_data,100)
                return explainer, global_shap_values, data_with_shap

    def add_shap_row(self, input_data, row_number):
        if is_classification(self.model):
            explainer, pred, pred_prob = self.shap_explainer()
        else:
            explainer, pred = self.shap_explainer()

        if type(explainer) == shap.explainers._tree.Tree:
            shap_values = explainer.shap_values(input_data)
            

            shap_row = self.append_shap_values_to_df(input_sv = shap_values[0], 
                                                           in_data= input_data.copy(),
                                                             scope='local') 
                         
            shap_row['Model Decision'] = pred(pd.DataFrame(input_data))[0]
            shap_row['Actual Decision'] = self.actual_data[row_number]
             
            if is_classification(self.model):
                probabilities = pred_prob(np.array(input_data))
                for i in range(len(np.unique(self.actual_data))):
                    shap_row['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]

            return shap_row
        else:
            shap_values = explainer.shap_values(input_data)
    
            shap_row = self.append_shap_values_to_df(input_sv = shap_values, 
                                                           in_data= input_data.copy(),
                                                             scope='local')

            shap_row['Model Decision'] = pred(pd.DataFrame(input_data).T)[0]
            if is_classification(self.model):
                probabilities = pred_prob([np.array(input_data)])
                for i in range(len(np.unique(self.actual_data))):
                    shap_row['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]
            return shap_row


    def shap_local(self, row_number):
        if is_classification(self.model):
            explainer, pred, pred_prob = self.shap_explainer()
            
        else:
            explainer, pred = self.shap_explainer()
            
        if row_number > len(self.input_data):
            raise IndexError(f"index {row_number} is out of bounds for axis 0 with size {len(self.input_data)}")
        else:
            if type(explainer) == shap.explainers._tree.Tree:
                local_shap_values = explainer.shap_values(self.input_data.iloc[row_number,:])
                row_with_shap = self.append_shap_values_to_df(input_sv = local_shap_values, 
                                                           in_data=self.input_data.iloc[row_number,:].copy(),
                                                             scope='local')

                row_with_shap['Model Decision'] = pred(pd.DataFrame(self.input_data.iloc[row_number]).T)[0]
                row_with_shap['True Values'] = pd.DataFrame(self.actual_data).iloc[row_number][0]
                
                if is_classification(self.model):
                    probabilities = pred_prob([np.array(self.input_data)[row_number]])
                    for i in range(len(np.unique(self.actual_data))):
                        row_with_shap['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]
                
                return explainer, local_shap_values, row_with_shap
            else:
                local_shap_values = explainer.shap_values(self.input_data.iloc[row_number,:])
                row_with_shap = self.append_shap_values_to_df(input_sv = local_shap_values, 
                                                           in_data= self.input_data.iloc[row_number,:].copy(),
                                                             scope='local')
                
                row_with_shap['Model Decision'] = pred(pd.DataFrame(self.input_data.iloc[row_number]).T)[0]
                row_with_shap['True Values'] = pd.DataFrame(self.actual_data).iloc[row_number][0]

                if is_classification(self.model):
                    probabilities = pred_prob(pd.DataFrame(self.input_data.iloc[row_number]).T)
                    for i in range(len(np.unique(self.actual_data))):
                        row_with_shap['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]

                return explainer, local_shap_values, row_with_shap


    def data_for_shap(self, input_data):
        if is_classification(self.model):
            explainer, pred, pred_fcn = self.shap_explainer()
            if type(explainer) == shap.explainers._tree.Tree:
                global_shap_values = explainer.shap_values(input_data)
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values[0], 
                                                                in_data=input_data.copy(), scope="local")
                prediction = pred([input_data])
                probabilities = pred_fcn([input_data])
                
                data_with_shap['Model Decision'] = prediction[0]
                #data_with_shap['True Values'] = self.actual_data
                

                for i in range(len(np.unique(self.actual_data))):
                    data_with_shap['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]
                return data_with_shap
            else:
                predictions = pred(shap.sample(input_data,100))
                global_shap_values = explainer.shap_values(shap.sample(input_data,100))  
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values[0], 
                                                               in_data=shap.sample(input_data,100).copy(),
                                                              scope='local')  
                prediction = pred(shap.sample(input_data,100))
                probabilities = pred_fcn(shap.sample(input_data,100))
                data_with_shap['Model Decision'] = prediction[0]
                #data_with_shap['True Values'] = self.actual_data
                
                for i in range(len(np.unique(self.actual_data))):
                    data_with_shap['Probability: {}'.format(np.unique(self.actual_data)[i])] = probabilities[:,i][0]

                return data_with_shap
        else:
            explainer, pred = self.shap_explainer()
            if type(explainer) == shap.explainers._tree.Tree:
                #Complete! Do not change. 
                global_shap_values = explainer.shap_values(input_data)
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values, 
                                                                in_data=self.input_data.copy(),
                                                                  scope="local")
                data_with_shap['Model Decision'] = pred(input_data)
                #data_with_shap['True Values'] = self.actual_data
                
                return data_with_shap

            else:
                global_shap_values = explainer.shap_values(shap.sample(input_data,100))  
                predictions = pred(shap.sample(input_data,100))
                data_with_shap = self.append_shap_values_to_df(input_sv = global_shap_values, 
                                                               in_data=shap.sample(input_data,100).copy(),
                                                              scope="local")
                data_with_shap['Model Decision'] = pred(shap.sample(self.input_data,100))
                #data_with_shap['True Values'] = self.actual_data
                
                return data_with_shap