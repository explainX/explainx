
from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff



class clf_performance():
    def __init__(self, y_true, y_pred):
        super(clf_performance, self).__init__()
        self.param = None
        self.y_true = y_true
        self.y_pred = y_pred
        
    def accuracy(self):
            # Acuracy_score
        accuracy = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)
        return accuracy
    
    
    def f1_score(self):
            # F1-score
        f1_score = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        return f1_score
    
    def recall_score(self):
            # Log Loss
        recall_score = sklearn.metrics.recall_score(self.y_true, self.y_pred)
        return recall_score
    
    def matthews_corrcoef(self):
            # Matthews Corrcoef
        matthews_corrcoef = sklearn.metrics.matthews_corrcoef(self.y_true, self.y_pred)
        return matthews_corrcoef
    
    def mae(self):
            # MAE
        mae = sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred)
        return mae
     
    def mse(self):
            # MSE
        mse = sklearn.metrics.mean_squared_error(self.y_true, self.y_pred)
        return mse
    
    
    def auc(self):
            # Area Under the Curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_true, self.y_pred)
        auc = metrics.auc(fpr, tpr)
        return auc
        
        
    def log_loss(self):
            # Log Loss
        log_loss = sklearn.metrics.log_loss(self.y_true, self.y_pred)
        return log_loss
        

    def matrix(self):
            # Confusion Matrix
        matrix = sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
        matrix_dataframe = pd.DataFrame(matrix)
        return matrix_dataframe
    
    
    def report(self):
             # Precision, Recall, F1-score, Support
        report = sklearn.metrics.classification_report(self.y_true, self.y_pred,output_dict=True)
        report_dataframe = pd.DataFrame(report)
        report_dataframe = report_dataframe.transpose().drop('accuracy', axis=0)
        return report_dataframe
    
    def performance_metrics(self):
       
            # Log Loss
        log_loss = sklearn.metrics.log_loss(self.y_true, self.y_pred)

            # Area Under the Curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_true, self.y_pred)
        auc = metrics.auc(fpr, tpr)

            # MAE
        mae = sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred) 
            # MSE
        mse = sklearn.metrics.mean_squared_error(self.y_true, self.y_pred)
         
            
        accuracy = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)   
            
        
        matthews_corrcoef = sklearn.metrics.matthews_corrcoef(self.y_true, self.y_pred)
    
       
        # Make DataFrames
        metric = ['Accuracy','Cross-Entropy Loss','Area Under Curve','MAE','MSE','Matthews Corrcoef']
        values = [accuracy, log_loss, auc, mae, mse,matthews_corrcoef]
        performance_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        
        
   
        return performance_dataframe
        
    
    def plot_metrics(self):
  
        performance_data = clf_performance(self.y_true, self.y_pred).performance_metrics()
        # Initialize a figure with ff.create_table(table_data)
        fig_metrics = ff.create_table(performance_data, height_constant=60)


        # Use the hovertext kw argument for hover text
        fig_metrics = go.Figure(data=[go.Bar(x=performance_data['metric'], y= performance_data['values'],
                    hovertext=[])])
        # Customize aspect
        fig_metrics.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig_metrics.update_layout(title_text='Classification Metrics')
        return fig_metrics  
    
    
    def plot_metrics_table(self):

   
        performance_data = clf_performance(self.y_true, self.y_pred).performance_metrics()
        fig_metrics_table = ff.create_table(performance_data, height_constant=60)
        
        return fig_metrics_table
    

    def plot_report(self):
       
        # Add table data
        report_data = clf_performance(self.y_true, self.y_pred).report()
        report_data.reset_index(inplace = True)
        # Initialize a figure with ff.create_table(table_data)
        fig_report = ff.create_table(report_data, height_constant=60)

         # Add graph data


        # Make traces for graph
        trace1 = go.Bar(x=report_data['index'], y=report_data['precision'], xaxis='x2', yaxis='y2',
                        marker=dict(color='#8A2BE2'),
                        name='prescision')
        trace2 = go.Bar(x=report_data['index'], y=report_data['recall'], xaxis='x2', yaxis='y2',
                        marker=dict(color='#000000'),
                        name='recall')

        trace3 = go.Bar(x=report_data['index'], y=report_data['f1-score'], xaxis='x2', yaxis='y2',
                        marker=dict(color='#00BFFF'),
                        name='f1score')

        # Add trace data to figure
        fig_report.add_traces([trace1, trace2,trace3])

        # initialize xaxis2 and yaxis2
        fig_report['layout']['xaxis2'] = {}
        fig_report['layout']['yaxis2'] = {}

        # Edit layout for subplots
        fig_report.layout.yaxis.update({'domain': [0, .45]})


        fig_report.layout.yaxis2.update({'domain': [.6, 1]})

        # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
        fig_report.layout.yaxis2.update({'anchor': 'x2'})
        fig_report.layout.xaxis2.update({'anchor': 'y2'})
        fig_report.layout.yaxis2.update({'title': 'Value'})


        # Update the margins to add a title and see graph x-labels.
        fig_report.layout.margin.update({'t':50, 'b':100})
        fig_report.layout.update({'title': 'Precision, Recall, f-1 Score'})

        # Update the height because adding a graph vertically will interact with
        # the plot height calculated for the table
        fig_report.layout.update({'height':900})

        return fig_report
    
    
    def plot_matrix(self):
        
        matrix_data =  clf_performance(self.y_true, self.y_pred).matrix()
        matrix_data = matrix_data.astype('float') / matrix_data.sum(axis=1)[:, np.newaxis]*100

        z = np.array(matrix_data).round(2)

        x = list(matrix_data.columns)
        y =  list(matrix_data.index)

        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]

        # set up figure 
        fig_matrix = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig_matrix.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          xaxis = dict(title='Predicted Label'),
                          yaxis = dict(title='True Label')
                         )





        # adjust margins to make room for yaxis title
        fig_matrix.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig_matrix['data'][0]['showscale'] = True
        return fig_matrix
    
    
    
    
class reg_performance():
    def __init__(self, y_true, y_pred):
        super(reg_performance, self).__init__()
        self.param = None
        self.y_true = y_true
        self.y_pred = y_pred
        
    def maxerror(self):
            # max_error metric calculates the maximum residual error.
        maxerror = sklearn.metrics.max_error(self.y_true, self.y_pred)
        return maxerror
    
    
    def mae(self):
            # Mean absolute error
        mae = sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred)
        return mae

    def mse(self):
            # Mean squared error regression
        mse = sklearn.metrics.mean_squared_error(self.y_true, self.y_pred)
        return mse
    
    
    def r2(self):
            # # R^2 (coefficient of determination) 
        r2 = sklearn.metrics.r2_score(self.y_true, self.y_pred)
        return r2
    
    
    def mape(self):
            # Mean absolute percentage error
            
            
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)    
        return mape   
            
  
    
  
    
    
   
    def performance_metrics(self):
        
        
        
        maxerror = sklearn.metrics.max_error(self.y_true, self.y_pred)

        mae = sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred)
      
        mse = sklearn.metrics.mean_squared_error(self.y_true, self.y_pred) 

        r2 = sklearn.metrics.r2_score(self.y_true, self.y_pred)
     
    
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        
        # Make DataFrames
        metric = ['Max Error','R squared','MAE','MSE','MAPE']
        values = [maxerror, r2, mae, mse, mape]
        metrics_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        
        return metrics_dataframe

    
    def plot_metrics(self):
        
        
       
     
        performance_data = reg_performance(self.y_true, self.y_pred).performance_metrics()
        # Initialize a figure with ff.create_table(table_data)
        fig_metrics = ff.create_table(performance_data, height_constant=60)
        

        # Use the hovertext kw argument for hover text
        fig_metrics = go.Figure(data=[go.Bar(x=performance_data['metric'], y= performance_data['values'],
                    hovertext=[])])
        # Customize aspect
        fig_metrics.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig_metrics.update_layout(title_text='Classification Metrics')
        
        return fig_metrics
    
    
    def plot_metrics_table(self):
        
        
       
     
        performance_data = reg_performance(self.y_true, self.y_pred).performance_metrics()
        fig_metrics_table = ff.create_table(performance_data, height_constant=60)
        
        return fig_metrics_table
    
        
        
  