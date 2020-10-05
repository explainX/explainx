# Import modules
import pandas as pd
import numpy as np
import sklearn
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import * 
from itertools import cycle
import colorlover as cl
import plotly.figure_factory as ff
import plotly.graph_objects as go

class performance_metrics():
    def __init__(self, y_test, y_pred):
        super(performance_metrics, self).__init__()
        self.param = None
        self.y_test = y_test
        self.y_pred = y_pred


    def false_possitives_negatives(self):
        
        f1 = f1_score(self.y_test, self.y_pred, average='micro')
        accuracy = accuracy_score(self.y_test, self.y_pred)
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float).sum()
        FN = FN.astype(float).sum()
        TP = TP.astype(float).sum()
        TN = TN.astype(float).sum()

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy for each class
        ACC = (TP+TN)/(TP+FP+FN+TN)


        fp_fn_table = pd.DataFrame(dict({'False Possitives (%)': [(FPR * 100).round(2)],
                                         'False Negatives (%)': [(FNR * 100).round(2)],
                                         'Accuracy (%)': accuracy.round(2)* 100,
                                         'F1': f1.round(2)}))
        
        fig_metrics_table = ff.create_table(fp_fn_table, height_constant=15)
        
        return fig_metrics_table    
    
        
        
    
    def get_matrix(self):
        y_test = label_binarize(self.y_test, classes=list(set(self.y_test.flatten())))
        n_classes = y_test.shape[1]
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), self.y_pred.argmax(axis=1))
        
        return conf_matrix 
    
    
    def plot_roc(self):
    
        y_test = label_binarize(self.y_test, classes=list(set(self.y_test.flatten())))
        n_classes = y_test.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        data = []
        trace1 = go.Scatter(x=fpr["micro"], y=tpr["micro"], mode='lines', 
                            line=dict(color='deeppink', width=2, dash='dot'),
                            name='micro-average ROC curve (area = {0:0.2f})'
                                   ''.format(roc_auc["micro"]))
        data.append(trace1)

        trace2 = go.Scatter(x=fpr["macro"], y=tpr["macro"], mode='lines', 
                            line=dict(color='navy', width=2, dash='dot'),
                            name='macro-average ROC curve (area = {0:0.2f})'
                                  ''.format(roc_auc["macro"]))
        data.append(trace2)

        colors = cycle(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

        for i, color in zip(range(n_classes), colors):
            trace3 = go.Scatter(x=fpr[i], y=tpr[i],
                                mode='lines', 
                                line=dict(color=color, width=2),
                                name='ROC curve of class {0} (area = {1:0.2f})'
                                ''.format(i, roc_auc[i]))
            data.append(trace3)

        trace4 = go.Scatter(x=[0, 1], y=[0, 1], 
                            mode='lines', 
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=False)


        layout = go.Layout(title='Receiver operating characteristic',
                           xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate'),
                           margin=dict(pad=25))

        fig = go.Figure(data=data, layout=layout)

        return fig
    
    
    def plot_pr(self):
        
        colors = cycle(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

        # Binarize the output
        y_test = label_binarize(self.y_test, classes=list(set(self.y_test.flatten())))
        n_classes = y_test.shape[1]

        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], self.y_pred[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], self.y_pred[:, i])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
            self.y_pred.ravel())
        average_precision["micro"] = average_precision_score(y_test, self.y_pred,
                                                             average="micro")


        data = []
        trace2 = go.Scatter(x=recall["micro"], y=precision["micro"], 
                            mode='lines',
                            line=dict(color='gold', width=2),
                            name='micro-average Precision-recall curve (area = {0:0.2f})'
                                  ''.format(average_precision["micro"]))
        data.append(trace2)
        
        for i, color in zip(range(n_classes), colors):
            trace3 = go.Scatter(x=recall[i], y=precision[i],
                                mode='lines',
                                line=dict(color=color, width=2),
                                name='Precision-recall curve of class {0} (area = {1:0.2f})'
                                      ''.format(i, average_precision[i]))
            data.append(trace3)

        layout = go.Layout(title='Precision-Recall curve',
                           xaxis=dict(title='Recall'),
                           yaxis=dict(title='Precision'),
                           margin=dict(pad=25))

        fig = go.Figure(data=data, layout=layout)
        
        return fig
    
    
    def plot_pie(self):
        
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        fp = FP.astype(float).sum()
        fn = FN.astype(float).sum()
        tp = TP.astype(float).sum()
        tn = TN.astype(float).sum()
        
        
        label_text = ["True Positive", "False Negative","False Positive","True Negative"]
        labels = ["TP", "FN", "FP", "TN"]
        blue = cl.flipper()['seq']['9']['Blues']
        red = cl.flipper()['seq']['9']['Reds']
        colors = [blue[4], blue[1], red[1], red[4]]

        trace0 = go.Pie(
        labels=label_text,
        values=[tp, fn, fp, tn],
        hoverinfo='label+value+percent',
        textinfo='text+value',
        text=labels,
        sort=False,
        marker=dict(
        colors=colors))

        layout = go.Layout(
        title=f'TP, TN, FP, FN',
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
        bgcolor='rgba(255,255,255,0)',
        orientation='h'))

        data = [trace0]
        figure = go.Figure(data=data, layout=layout)
        return figure
    
    
    def plot_matrix(self):
        
        matrix_data =  pd.DataFrame(confusion_matrix(self.y_test, self.y_pred))
        matrix_data = matrix_data.astype('float') / matrix_data.sum(axis=1)[:, np.newaxis]*100

        z = np.array(matrix_data).round(2)

        x = list(matrix_data.columns)
        y =  list(matrix_data.index)

        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]

        # set up figure 
        fig_matrix = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig_matrix.update_layout(title_text='<i><b>Confusion matrix (%)</b></i>',
                          xaxis = dict(title='Predicted Label'),
                          yaxis = dict(title='True Label'))

        # adjust margins to make room for yaxis title
        fig_matrix.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig_matrix['data'][0]['showscale'] = True
        
        return fig_matrix
    
    
    def metrics_table(self):
       
        
                 # AUC score
        lb = preprocessing.LabelBinarizer()
        lb.fit(self.y_test)
        y_test_roc = lb.transform(self.y_test)
        y_pred_roc = lb.transform(self.y_pred)
        auc = roc_auc_score(y_test_roc, y_pred_roc, average="macro").round(2)

                    # MAE
        mae = mean_absolute_error(self.y_test, self.y_pred).round(2)
                # MSE
        mse = mean_squared_error(self.y_test, self.y_pred).round(2)

        # Accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred).round(2)   

        # matthews_corrcoef
        matthews_corrcoe = matthews_corrcoef(self.y_test, self.y_pred).round(2)

    
       
        # Make DataFrames
        metric = ['Accuracy','Area Under Curve','MAE','MSE','Matthews Corrcoef']
        values = [accuracy, auc, mae, mse,matthews_corrcoe]
        metrics_tab = pd.DataFrame({'metric': metric, 'values': values})
        
        fig_metrics_table = ff.create_table(metrics_tab, height_constant=15)
        
   
        return fig_metrics_table



    def performance_metrics_regression(self):
        
        
        
        maxerror = max_error(self.y_test, self.y_pred).round(2) 

        mae = mean_absolute_error(self.y_test, self.y_pred).round(2) 
      
        mse = mean_squared_error(self.y_test, self.y_pred).round(2) 

        r2 = r2_score(self.y_test, self.y_pred).round(2) 
     
    
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(self.y_test, self.y_pred).round(2) 
    
        
        # Make DataFrames
        metric = ['Max Error','R squared','MAE','MSE','MAPE']
        values = [maxerror, r2, mae, mse, mape]
        metrics_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        metrics_dataframe = ff.create_table(metrics_dataframe, height_constant=15)
        return metrics_dataframe

    def actuals(self):
        
        
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        fp = FP.astype(float).sum().round()
        fn = FN.astype(float).sum().round()
        tp = TP.astype(float).sum().round()
        tn = TN.astype(float).sum().round()

        total = tn + fn + fp + tp

        predicted_no = [tn,fn]
        predicted_yes = [fp,tp]


        data = {'Predicted No':predicted_no, 'Predicted Yes':predicted_yes} 

        # Creates pandas DataFrame. 
        df = pd.DataFrame(data, index =['No', 'Yes'])
        df['Total'] = df.sum(axis=1)
        df.loc['Total',:]= df.sum(axis=0)
        df.reset_index(inplace=True)
        df.rename({"index" : 'Actual'}, axis=1, inplace=True)

        actuals_table = ff.create_table(df, height_constant=15)

        return actuals_table