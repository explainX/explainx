from imports import *

class cohortAnalysis():
    def __init__(self):
        self.cohorts = {}
        self.cohort_metrics = []
        self.cohort_set = {}
        
        
    def filtered_dataframe(self, df, filter_variable, var_name="", operator="", value=""):
        """
        data = main_data 
        name = cohort_name 
        var_name = name of the variable to slice/dice 
        operator: >, <, =, >=, <= 
        value = value of the variable 

        returns main_data: filtered dataset with just the probabilities
                name: filtered dataset with the condition
        """
        main_dataset = df[filter_variable]
        if (var_name != "") or (operator != "") or (value != ""):
            if len(df[filter_variable]) != 0:
                name =  df.query("{} {} {}".format(var_name, operator, value))[filter_variable]
                condition = str(var_name)+str(operator)+str(value)
                return main_dataset, name, condition
            else:
                pass
        else: 
            if len(df[filter_variable]) != 0:
                condition = "All Data"
                return main_dataset, condition
            else:
                pass
    
    def add_cohort(self, df, filter_variable, var_name="", operator="", value=""):
        if (var_name != "") or (operator != "") or (value != ""):
                main_dataset, name, condition = self.filtered_dataframe(df,filter_variable,var_name,operator,value)
                self.cohorts[condition] = name
        else:
                main_dataset, condition = self.filtered_dataframe(df, filter_variable )
                self.cohorts[condition] = main_dataset

    def remove_cohort(self):
        if (len(self.cohorts) >1) and (len(self.cohort_set) > 1):
            self.cohorts.popitem()
            self.cohort_set.popitem()
        else:
            pass
            
    def add_cohort_metrics(self, df, var_name="", operator="", value="", is_classification=True):
        """
        data = main_data 
        name = cohort_name 
        var_name = name of the variable to slice/dice 
        operator: >, <, =, >=, <= 
        value = value of the variable  

        """
        if value != "":
            #Extract filtered predicted values
            _, predicted, condition_predict = self.filtered_dataframe(df, "y_prediction",var_name,operator,value)
            #Extract filtered true labels
            _, true_values, condition_true = self.filtered_dataframe(df, "y_actual", var_name, operator, value)
            #calculate metrics
            if is_classification is True:
                if len(true_values) != 0:
                    accuracy, precision, recall, fpr, fnr = self.classification_cohort_metrics(true_values, predicted)
                    self.cohort_set[condition_predict] = self.generate_classification_divs(accuracy, precision, recall, fpr, fnr)
                else:
                    pass
            else:
                if len(true_values) != 0:
                    mae, mse, r2 = self.regression_cohort_metrics(true_values, predicted)
                    #save these metrics to an array
                    self.cohort_set[condition_predict] = self.generator_regression_divs(mae, mse, r2)
                else:
                    pass
        else:
            main_dataset, condition = self.filtered_dataframe(df, "y_prediction")
            true_data, _ = self.filtered_dataframe(df, "y_actual")
            if is_classification is True:
                if len(true_data) != 0:
                    accuracy, precision, recall, fpr, fnr = self.classification_cohort_metrics(true_data,main_dataset)
                    self.cohort_set[condition] =  self.generate_classification_divs(accuracy, precision, recall, fpr, fnr)
                else:
                    pass
            else:
                if len(true_data) != 0:
                    mae, mse, r2 = self.regression_cohort_metrics(true_data, main_dataset)
                    #save these metrics to an array
                    self.cohort_set[condition] = self.generator_regression_divs(mae, mse, r2)
                else:
                    pass

    def generate_classification_divs(self, accuracy, precision, recall, fpr, fnr):
        metrics_div = [html.Div("Accuracy: {}".format(accuracy)),
                          html.Div("Precision: {}".format(precision)),
                          html.Div("Recall: {}".format(recall)),
                          html.Div("fpr: {}".format(fpr)),
                          html.Div("fnr: {}".format(fnr))
                         ]
        return metrics_div
    
    def generator_regression_divs(self, mae, mse, r2):
        metrics_div = [html.Div("MAE : {}".format(mae)),
                          html.Div("MSE : {}".format(mse)),
                          html.Div("R2: {}".format(r2))]
        return metrics_div

    def cohort_details(self):
        """
        Cohort Name
        Length of Cohort
        """
        length_dict = {key: len(value) for key, value in self.cohorts.items()}
        divs = []
        for i in range(len(length_dict)):
            if list(length_dict.values())[i] != 0:
                first_html = html.Div(list(length_dict.keys())[i])
                second_html = html.Div(str(list(length_dict.values())[i])+" datapoints")
                divs.append(html.Div([first_html,second_html], style={"padding-left":"20px","padding-right":"20px","padding-bottom":"0px","width":"200px"}))
            else:
                pass
        return divs
        
    def cohort_metrics_details(self):
        """
        Cohort Name
        Metrics
        """
        length_dict = {key: value for key, value in self.cohort_set.items()}
        div_metrics = []
        for i in range(len(length_dict)):
            div_metrics.append(html.Div(list(length_dict.values())[i], style={"padding-left":"20px","padding-right":"20px","padding-bottom":"0px","width":"200px"}))
        return div_metrics
    
   
    def cohort_graph(self, filter_variable):
        """[This function generators the box plot for the cohorts. This is operated directly from the frontend.]

        Args:
            filter_variable ([string]): [This variable is x-axis value of the graph. It can be either probabilities or model prediction values]

        Returns:
            [figure]: [box plot graph]
        """
        
        X_Value = str(filter_variable)
        Y_Value = 'Cohorts'
        
        fig = go.Figure()

        for k, v in self.cohorts.items():
            fig.add_trace(go.Box(x=v, name=k))

        fig.update_layout(
            yaxis_title = Y_Value,
            xaxis_title = X_Value,
            template = "plotly_white",
            font=dict(
                size=8,
            )
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        fig.update_layout(
            margin={"t":0},
        )
        return fig
    
    def classification_cohort_metrics(self, y_true, predicted): 
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
        if len(y_true) != 0:
            #Accuracy
            accuracy = round(metrics.accuracy_score(y_true, predicted),2) #Accuracy classification score.
            #Precision
            precision = round(metrics.precision_score(y_true, predicted, zero_division=1),2) #Compute the precision
            #Recall
            recall = round(metrics.recall_score(y_true, predicted, zero_division=1),2) #Compute the recall
            #False Positive Rate (FPR)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, predicted).ravel() #Compute confusion matrix to evaluate the accuracy of a classification.
            #False Negative Rate (FNR)
            fpr = round((fp/(fp+tn)),2)
            fnr = round((fn/(tp+fn) if (tp+fn) else 0),2) 

            return accuracy, precision, recall, fpr, fnr
        else:
            pass
    
    def regression_cohort_metrics(self, y_true, predicted):
        """[Calculates the metrics for regression problems]

        Args:
            y_true ([type]): [True labels from the dataset]
            predicted ([type]): [Predicted values from the model]

        Returns:
            Mean Absolute Error
            Mean Squared Error
            R-Squared Value
        """
        if len(y_true) != 0:
            #Mean Absolute Error
            mae = round(metrics.mean_absolute_error(y_true, predicted),2)
            #Mean Squared Error
            mse = round(metrics.mean_squared_error(y_true, predicted),2)
            #R2
            r2 = round(metrics.r2_score(y_true, predicted),2)
            
            return mae, mse, r2
        else:
            pass


    
    