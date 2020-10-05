


class calculate_metrics():
    def __init__(self):
        super(calculate_metrics, self).__init__()
        self.param = None

    def classification_metrics(self, y_test, model_predict):
       
        # Acuracy_score
        acuracy_score = metrics.accuracy_score(y_test, model_predict) 

        # Log Loss
        y_probs = model.predict_proba(X_test)
        LogLoss = log_loss(y_test, y_probs, labels=model.classes_)

        # Explained Variance Score
        explained_variancescore = explained_variance_score(y_test, model_predict)


        fpr, tpr, thresholds =metrics.roc_curve(y_test, model_predict, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        # MAE
        mae = mean_absolute_error(y_test, model_predict) 
        # MSE
        mse = mean_squared_error(y_test, model_predict)
        # RMS
        rms = sqrt(mse)   

        # Precision, Recall, F1-score, Support
        report = classification_report(y_test, model_predict,output_dict=True)
        report_dataframe = pd.DataFrame(report)
        report_dataframe = report_dataframe.transpose()

        # Confusion Matrix
        matrix = confusion_matrix(y_test, model_predict)
        matrix_dataframe = pd.DataFrame(matrix)

        # Make DataFrames
        metric = ["Accuracy Score",'Cross-Entropy Loss','Area Under Curve','MAE','MSE','RMS']
        values = [acuracy_score, LogLoss, auc, mae, mse, rms]
        metrics_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        metrics_dataframe.set_index('metric', inplace = True)

    
   
        return metrics_dataframe, report_dataframe, matrix_dataframe
     
    
    
    
    def regression_metrics(self, y_test, model_predict):
       
        # #Explained variance regression score 
        exp_variance_score = explained_variance_score(y_test, model_predict)

        #max_error metric calculates the maximum residual error.
        maxerror = max_error(y_test, model_predict)

        #Mean absolute error regression loss
        mae = mean_absolute_error(y_test, model_predict) 

        #Mean squared error regression loss
        mse = mean_squared_error(y_test, model_predict) 

        # RMSE
        rmse = sqrt(mse)

        # R^2 (coefficient of determination) regression score function.
        r2 = r2_score(y_test, model_predict) 

        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(y_test, model_predict)

        # Make DataFrames
        metric = ["Explained Variance Score",'Max Error','R squared','MAE','MSE','RMSE','MAPE']
        values = [exp_variance_score, maxerror, r2, mae, mse, rmse, mape]
        metrics_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        metrics_dataframe.set_index('metric', inplace = True)

        return metrics_dataframe
