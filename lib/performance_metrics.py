
from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import colorlover as cl


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
        metric = ['Accuracy','Area Under Curve','MAE','MSE','Matthews Corrcoef']
        values = [accuracy, auc, mae, mse,matthews_corrcoef]
        performance_dataframe = pd.DataFrame({'metric': metric, 'values': values})
        
        
   
        return performance_dataframe
        
    def y_score(self, model, X_test):
        y_score = model.predict_proba(X_test)[:,1]
        
        return y_score
    
    def fp_fn_table(self):
        tn, fp, fn, tp = np.array(clf_performance(self.y_true, self.y_pred).matrix()).ravel()

        f1 = clf_performance(self.y_true, self.y_pred).f1_score()
        accuracy = clf_performance(self.y_true, self.y_pred).accuracy()

        total = fp + fn + tp + tn

        fp_fn_table = pd.DataFrame(dict({'False Possitives (%)': [(fp / total * 100).round(2)],
             'False Negatives (%)': [(fn / total * 100).round(2)],
             'Accuracy (%)': accuracy.round(2),
             'F1': f1.round(2)}))
        
        return fp_fn_table   
        
        
        
    def F_pos_F_neg(self):
    
        tn, fp, fn, tp = np.array(clf_performance(self.y_true, self.y_pred).matrix()).ravel()

        predicted_yes = [tp,fp]
        predicted_no = [fn,tn]

        index_names = ['Actual Yes', 'Actual No']
        total = predicted_yes + predicted_no

        df = pd.DataFrame({'Predicted Yes': predicted_yes,
                     'Predicted No': predicted_no}, index = index_names)


        df['Total'] = df['Predicted Yes'] + df['Predicted No']
        df.loc['Total'] = df.sum(axis=0) 


        df['Predicted Yes2']  = (df['Predicted Yes'] / df['Total'] *100).round(1)
        df['Predicted No2']  = (df['Predicted No'] / df['Total'] * 100).round(1)
        total_sum = df.Total.iloc[:-1].sum()
        df['Total2'] = (df['Total'] / total_sum * 100).round(1)

        PY = [f'{str(j)}{"%"} ({str(x)}) ' for x, j in zip(df['Predicted Yes'], df['Predicted Yes2'])]
        PrN = [f'{str(j)}{"%"} ({str(x)}) ' for x, j in zip(df['Predicted No'], df['Predicted No2'])]
        Tt = [f'{str(j)}{"%"} ({str(x)}) ' for x, j in zip(df['Total'], df['Total2'])]

        df['Predicted Yes'] = PY
        df['Predicted No'] = PrN
        df['Total'] = Tt
        df['*'] = ['Actual Yes', 'Actual No', 'Total']

        RP_FN = df[['*','Predicted Yes', 'Predicted No', 'Total']]
        RP_FN.style.applymap('font-weight: bold', subset=['Index'])

        return RP_FN
    
    
    def plot_roc_curve(self, model, X_test):
        decision_test = clf_performance(self.y_true, self.y_pred).y_score(model, X_test)
        fpr, tpr, threshold = metrics.roc_curve(self.y_true, decision_test)

            # AUC Score
        auc_score = metrics.roc_auc_score(y_true=self.y_true, y_score=decision_test)

        trace0 = go.Scatter(x=fpr,y=tpr,name='ROC',)
        layout = go.Layout(title=f'ROC Curve (AUC = {auc_score:.3f})',xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'))

        figure = go.Figure(data=trace0, layout=layout)

        return figure

    def plot_precision_recall(self, model, X_test):
        y_score=clf_performance(self.y_true, self.y_pred).y_score(model, X_test)
        decision_test = clf_performance(self.y_true, self.y_pred).y_score(model, X_test)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(self.y_true, y_score)
        auc_score = metrics.roc_auc_score(y_true=self.y_true, y_score=decision_test)

        trace0 = go.Scatter(x=recall,y=precision,name='ROC',)


        layout = go.Layout(title=f'PR Curve (AUC = {auc_score:.3f})',xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision'))

        figure = go.Figure(data=trace0, layout=layout)



        return figure

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
    
    def plot_pie(self):
        
        tn, fp, fn, tp = np.array(clf_performance(self.y_true, self.y_pred).matrix()).ravel()

        label_text = ["True Positive",
                          "False Negative",
                          "False Positive",
                          "True Negative"]
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
        colors=colors
        )
        )

        layout = go.Layout(
        title=f'TP, TN, FP, FN',
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
        bgcolor='rgba(255,255,255,0)',
        orientation='h'
        )
        )

        data = [trace0]
        figure = go.Figure(data=data, layout=layout)
        return figure
    
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
    
        
        
  