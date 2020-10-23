from imports import *
from feature_impact import *
from feature_importance import *
from shap_pdp import *
from summary_plot import *
import dash_daq as daq
from data_for_shap_graphs import *

class what_if():
    def __init__(self):
        super(what_if, self).__init__()
        self.data = data_for_shap_graphs()

        # save all important variables here.
    def col_values_pairs(self, df, features):
        df_features= df[features]
        df_describe= df_features.describe()
        
        """
        which features are categorical?
        find their feature value pairs
        """

        numnerical_col= df_describe.columns
        categorical_col= list(set(features)- set(numnerical_col))

        col_type_dict={}
        col_value_dict={}
        for f in numnerical_col:
            col_type_dict[f]="slider"
            col_value_dict[f]= self.get_min_max_featurs(df_describe, f)
            
        
        for f in categorical_col:
            categ,col_type= all_categories(df, f)
                
            col_type_dict[f]= col_type
            col_value_dict[f]= categ
            
        return col_value_dict, col_type_dict


    def all_categories(self, df, feature_name):
        
        """
        find all categories in categorical features.
        """
        all_categories=list(set(df[feature_name]))
        
        return all_categories, 'dropdown'

        
    def get_min_max_featurs(self, df_describe, feature_name):

        x_des_var_list=list(df_describe[feature_name])
        q00= x_des_var_list[3]-25/100*x_des_var_list[3]
        q0= x_des_var_list[3]
        q1= x_des_var_list[4]
        q2= x_des_var_list[5]
        q3= x_des_var_list[6]
        q4= x_des_var_list[7]
        q5= x_des_var_list[7]+ 25/100*x_des_var_list[7]
        ans= [q00,q0,q1,q2,q3,q4,q5]
        
        # round off to 2 dp
        for i in range(len(ans)):
            ans[i]= round(ans[i],2)
        
        
        return ans

    def what_if_form(self, df, features):
        col_value_dict, col_type_dict = self.col_values_pairs(df, features)
        
        form_group_array=[]
        for f in features:
            """
            if feature type== slider
            then following is correct.
            """
            if col_type_dict[f]=='slider':
                #form_group_array.append(self.form_group_slider(f, col_value_dict[f]))
                

                form_group_array.append(self.form_group_input(f, col_value_dict[f]))
                
            elif col_type_dict[f]=='radio':
                
                form_group_array.append(self.form_group_radio(f, col_value_dict[f]))
            else:
                
                form_group_array.append(self.form_group_dropdown(f, col_value_dict[f]))
                                    
                                    
        return html.Form(form_group_array)
            
    def form_group_input(self, feature, values, value=None):
        marks= {}
        for v in values:
            marks[v]=str(v)
        
        if value==None:
            value=values[3]
        
        fg= dbc.FormGroup([
                    html.Div(feature, className="form_feature"),
                    html.Div(dcc.Input(
                        className = "form_group_input",
                        id=feature+'_slider',
                        type="number",
                        value = value,
                        debounce=True,
                        placeholder="Enter the value"))], 
                        className="what_if_form_group")
        
        return fg
        
        
    def form_group_slider(self, feature, values, value=None):
        marks= {}
        for v in values:
            marks[v]=str(v)
        
        if value==None:
            value=values[3]
        
        fg= dbc.FormGroup([
                    html.Div(feature, className="form_feature"),
                    html.Div(daq.Slider(
                                className = "form_group_slider",
                                id=feature+'-slider',
                                min=values[0],
                                max=values[-1],
                            handleLabel={"showCurrentValue": True,"label": 'Value'}))], className="what_if_form_group")
        
        return fg


    def form_group_dropdown(self, feature, values, value=None):
        options= []
        for v in values:
            options.append({'label': v, 'value': v})
        
        if value==None:
            value=values[0]
        
        fg = dbc.FormGroup([
                    html.Div(feature, className="form_feature"),
                    html.Div(dcc.Dropdown(
                                className = "form_group_dropdown",
                                id= feature+'-slider',
                                options=options,
                                value=value))],className="what_if_form_group")
        
        return fg

