from imports import *
from rescale_numeric_feature import *



"""
This class calculates feature importance

Input: 


"""


class calculate_shap():
    def __init__(self):
        super(calculate_shap, self).__init__()
        self.param = None

    def xgboost_shap(self, model, X):
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        pd_shap = pd.DataFrame(shap_values)
        all_columns = list(X.columns)

        shap_columns = []

        for i in all_columns:
            shap_columns.append(i + "_impact")

        pd_shap.columns = shap_columns

        Y = X.copy()
        for c in shap_columns:
            Y[c] = list(pd_shap[c])

        return Y, explainer

    def catboost_shap(self, model, df, y_variable=None):
        import catboost
        from catboost import CatBoostClassifier, Pool
        from catboost import CatBoostRegressor
        from catboost.utils import get_confusion_matrix
        # explain the model's predictions using SHAP
        if y_variable != None:
            try:
                df = df.drop(y_variable, axis=1)
            except:
                df = df

        # find categorical variables and it's index
        g = get_cols()

        cat, cat_index = g.cate_col_with_index(df)
        all_columns = list(df.columns)

        # convert dataframe in to array
        df_array = df.to_numpy()

        # call the function
        shap_values = self.get_shap_values(df_array, model, all_columns, cat_index)

        # append the results with the original file

        # final_df = df.join(shap_values)

        shap_columns= shap_values.columns

        Y = df.copy()
        for c in shap_columns:
            Y[c] = list(shap_values[c])


        return Y


    def kernel_shap(self, model, X_train):
        # use Kernel SHAP to explain test set predictions
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_train, nsamples=100)

        pd_shap = pd.DataFrame(shap_values)
        all_columns = list(X_train.columns)

        shap_columns = []

        for i in all_columns:
            shap_columns.append(i + "_impact")
        pd_shap.columns = shap_columns



        Y = X_train.copy()
        for c in shap_columns:
            Y[c] = list(pd_shap[c])

        return Y, explainer

    def kernel_shap_classification(self, model, X_train,prediction_col):
        # use Kernel SHAP to explain test set predictions
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_train, nsamples=100)

        pd_shap = self.select_row_shap_values(shap_values, prediction_col)
        all_columns = list(X_train.columns)

        shap_columns = []

        for i in all_columns:
            shap_columns.append(i + "_impact")
        pd_shap.columns = shap_columns



        Y = X_train.copy()
        for c in shap_columns:
            Y[c] = list(pd_shap[c])

        return Y, explainer

    def select_row_shap_values(self, shap_values,prediction_col):

        num_of_classes = len(shap_values)

        if num_of_classes== len(prediction_col):
            df_final = pd.DataFrame(shap_values)
            return df_final

        point_no=0
        df_array = []
        for p in prediction_col:
            df_array.append(shap_values[p][point_no])
            point_no=point_no+1

        df_final = pd.DataFrame(df_array)
        return df_final


    def randomforest_shap_classification(self, model, X,prediction_col):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X,approximate=True)


        pd_shap = self.select_row_shap_values(shap_values,prediction_col)
        all_columns = list(X.columns)


        pd_shap.columns = [f"{y}_impact" for y in all_columns]

        shap_columns = pd_shap.columns

        Y = X.copy()
        for c in shap_columns:
            Y[c] = list(pd_shap[c])


        return Y, explainer


    def randomforest_shap(self, model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X,approximate=True)


        pd_shap = pd.DataFrame(shap_values)
        all_columns = list(X.columns)


        pd_shap.columns = [f"{y}_impact" for y in all_columns]

        shap_columns = pd_shap.columns

        Y = X.copy()
        for c in shap_columns:
            Y[c] = list(pd_shap[c])


        return Y, explainer


    def get_shap_values(self, x_array, model, x_variable, cat_index):
        """
        SHAP VALUES CALCULATED
        """
        shap_values = model.get_feature_importance(Pool(x_array, cat_features=cat_index), type='ShapValues')
        shap_values = shap_values[:, :-1]
        total_columns = x_variable
        total_columns = [i + '_impact' for i in total_columns]

        shap_values = pd.DataFrame(data=shap_values, columns=total_columns)
        return shap_values

    def find(self, model, df,prediction_col,is_classification, model_name="xgboost"):

        if model_name == "xgboost":
            df2 , explainer= self.xgboost_shap(model, df)
            return df2, explainer

        elif model_name == "lightgbm":
            df2 , explainer= self.xgboost_shap(model, df)

            return df2, explainer

        elif model_name == "catboost":
            df2 = self.catboost_shap(model, df)
            explainer= None
            return df2, explainer


        elif model_name == "randomforest":
            if is_classification:
                df2 , explainer= self.randomforest_shap_classification(model, df, prediction_col)
            else:
                df2 , explainer= self.randomforest_shap(model, df)
            return df2, explainer

        elif model_name == "svm":
            if is_classification:
                df2 , explainer= self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2, explainer = self.kernel_shap(model, df)
            return df2, explainer

        elif model_name == "knn":
            if is_classification:
                df2, explainer = self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2 , explainer= self.kernel_shap(model, df)
            return df2, explainer

        elif model_name == "logisticregression":
            if is_classification:
                df2 , explainer= self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2, explainer = self.kernel_shap(model, df)
            return df2, explainer

        elif model_name == "decisiontree":
            if is_classification:
                df2, explainer = self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2, explainer = self.kernel_shap(model, df)
            return df2, explainer

        elif model_name == "neuralnetwork":
            if is_classification:
                df2, explainer = self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2, explainer = self.kernel_shap(model, df)
            return df2, explainer

        elif model_name=="gradientboostingregressor":
            df2 , explainer= self.xgboost_shap(model, df)
            return df2, explainer
        elif "gradientboosting" in model_name:
            df2, explainer = self.xgboost_shap(model, df)
            return df2, explainer
        else:
            if is_classification:
                df2 , explainer= self.kernel_shap_classification(model, df,prediction_col)
            else:
                df2 , explainer= self.kernel_shap(model, df)
            return df2, explainer




