import os
import sys
#import re
from pathlib import Path
from sys import platform
import subprocess
#import time

path = Path(__file__).parent.absolute()
path_dataset = os.path.join(path, "datasets")
path = os.path.join(path, "lib")

sys.path.append(path)

from imports import *
from dashboard import *
from calculate_shap import *
from analytics import Analytics


class explain():
    def __init__(self):
        super(explain, self).__init__()
        self.param = {}

    # is classification function?

    # def is_classification_given_y_array(self, y_test):
    #     is_classification = False
    #     total = len(y_test)
    #     total_unique = len(set(y_test))
    #     if total < 30:
    #         if total_unique < 10:
    #             is_classification = True
    #     else:
    #         if total_unique < 20:
    #             is_classification = True
    #     return is_classification

    def random_string_generator(self):
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        return random_str

    def ai_h2o_automl(self, df, y_column_name, model, model_name="h2o", mode=None):
        y_variable = "y_actual"
        y_variable_predict = "y_prediction"
        y_variable = "y_actual"
        y_variable_predict = "y_prediction"
        instance_id = self.random_string_generator()
        analytics = Analytics()
        analytics['ip'] = analytics.finding_ip()
        analytics['mac'] = analytics.finding_address()
        analytics['instance_id'] = instance_id
        analytics['time'] = str(datetime.datetime.now())
        analytics['total_columns'] = len(df.columns)
        analytics['total_rows'] = len(df)
        analytics['os'] = analytics.finding_system()
        analytics['model_name'] = model_name
        analytics["function"] = 'before_dashboard'
        analytics["query"] = "before_dashboard"
        analytics['finish_time'] = ''
        analytics.insert_data()

        # If yes, then different shap functuions are required.
        # get the shap value based on predcton and make a new dataframe.

        # find predictions first as shap values need that.

        prediction_col = []

        if model_name == 'h2o':
            if isinstance(df, pd.DataFrame):
                df = h2o.H2OFrame(df)
            prediction_col = model.predict(df[y_column_name])
        # is classification?

        is_classification = True if model.type == 'classifier' else False
        # shap
        c = calculate_shap()
        self.df_final, self.explainer = c.find(model, df, prediction_col, is_classification,
                                               model_name=model_name)

        # prediction col
        self.df_final[y_variable_predict] = prediction_col.as_data_frame()[y_column_name].tolist()

        self.df_final[y_variable] = df.as_data_frame()[y_column_name].tolist()

        # additional inputs.
        if is_classification is True:
            # find and add probabilities in the dataset.
            try:
                prediction_col_prob = model.predict_proba(df)
            except:
                prediction_col_prob = model.predict(df)
            prediction_col_prob = prediction_col_prob.as_data_frame()

            pd_prediction_col_prob = pd.DataFrame(prediction_col_prob)

            for c in pd_prediction_col_prob.columns:
                self.df_final["probability_of_predicting_class_" + str(c)] = list(pd_prediction_col_prob[c])

            classes = []
            for c in pd_prediction_col_prob.columns:
                classes.append(str(c))
            self.param["classes"] = classes

            try:
                expected_values_by_class = self.explainer.expected_value
            except:
                expected_values_by_class = []
                for c in range(len(classes)):
                    expected_values_by_class.append(1 / len(classes))

            self.param["expected_values"] = expected_values_by_class
        else:
            try:
                expected_values = self.explainer.expected_value
                self.param["expected_values"] = [expected_values]
            except:
                expected_value = [round(np.array(y).mean(), 2)]
                self.param["expected_values"] = expected_value

        self.param["is_classification"] = is_classification
        self.param["model_name"] = model_name
        self.param["model"] = model
        self.param["columns"] = df.columns
        self.param["y_variable"] = y_variable
        self.param["y_variable_predict"] = y_variable_predict
        self.param['instance_id'] = instance_id

        d = dashboard()
        d.find(self.df_final, mode, self.param)

        return True

    def ai(self, df, y, model, model_name="xgboost", mode=None):
        y_variable = "y_actual"
        y_variable_predict = "y_prediction"

        # Code for Analytics
        instance_id = self.random_string_generator()
        analytics = Analytics()
        analytics['ip'] = analytics.finding_ip()
        analytics['mac'] = analytics.finding_address()
        analytics['instance_id'] = instance_id
        analytics['time'] = str(datetime.datetime.now())
        analytics['total_columns'] = len(df.columns)
        analytics['total_rows'] = len(df)
        analytics['os'] = analytics.finding_system()
        analytics['model_name'] = model_name
        analytics["function"] = 'before_dashboard'
        analytics["query"] = "before_dashboard"
        analytics['finish_time'] = ''
        analytics.insert_data()

        prediction_col = []

        if model_name == "xgboost":
            import xgboost
            if xgboost.__version__ in ['1.1.0', '1.1.1', '1.1.0rc2', '1.1.0rc1']:
                print(
                    "Current Xgboost version is not supported. Please install Xgboost using 'pip install xgboost==1.0.2'")
                return False
            prediction_col = model.predict(xgboost.DMatrix(df))

        elif model_name == "catboost":
            prediction_col = model.predict(df.to_numpy())

        else:
            prediction_col = model.predict(df)

        # is classification?
        # is_classification = self.is_classification_given_y_array(prediction_col)
        ModelType = lambda model: True if is_classifier(model) else False
        is_classification = ModelType(model)

        # shap
        c = calculate_shap()
        self.df_final, self.explainer = c.find(model, df, prediction_col, is_classification, model_name=model_name)

        # Append Model Decision & True Labels Columns into the dataset.
        self.df_final[y_variable_predict] = prediction_col
        self.df_final[y_variable] = y

        # additional inputs.
        if is_classification == True:
            # find and add probabilities in the dataset.
            # prediction_col_prob = model.predict_proba(df)
            # pd_prediction_col_prob = pd.DataFrame(prediction_col_prob)

            probabilities = model.predict_proba(df)

            for i in range(len(np.unique(prediction_col))):
                self.df_final['Probability: {}'.format(np.unique(prediction_col)[i])] = probabilities[:, i]

            self.param['classes'] = np.unique(prediction_col)

            # for c in pd_prediction_col_prob.columns:
            #   self.df_final["probability_of_predicting_class_" + str(c)] = list(pd_prediction_col_prob[c])

            # classes = []
            # for c in pd_prediction_col_prob.columns:
            #   classes.append(str(c))
            # self.param["classes"] = classes

            try:
                expected_values_by_class = self.explainer.expected_value
            except:
                expected_values_by_class = []
                for c in range(len(np.unique(prediction_col))):
                    expected_values_by_class.append(1 / len(np.unique(prediction_col)))

            self.param["expected_values"] = expected_values_by_class
        else:
            try:
                expected_values = self.explainer.expected_value
                self.param["expected_values"] = [expected_values]
            except:
                expected_value = [round(np.array(y).mean(), 2)]
                self.param["expected_values"] = expected_value

        self.param["is_classification"] = is_classification
        self.param["model_name"] = model_name
        self.param["model"] = model
        self.param["columns"] = df.columns
        self.param["y_variable"] = y_variable
        self.param["y_variable_predict"] = y_variable_predict
        self.param['instance_id'] = instance_id

        d = dashboard()
        d.find(self.df_final, mode, self.param)

        return True

    def ai_test(self, df, y, model, model_name="xgboost", mode=None):
        y_variable = "y_actual"
        y_variable_predict = "y_prediction"

        prediction_col = []

        if model_name == "xgboost":
            import xgboost
            if xgboost.__version__ in ['1.1.0', '1.1.1', '1.1.0rc2', '1.1.0rc1']:
                print(
                    "Current Xgboost version is not supported. Please install Xgboost using 'pip install xgboost==1.0.2'")
                return False
            prediction_col = model.predict(xgboost.DMatrix(df))

        elif model_name == "catboost":
            prediction_col = model.predict(df.to_numpy())

        else:
            prediction_col = model.predict(df.to_numpy())

        # is classification?
        is_classification = self.is_classification_given_y_array(prediction_col)

        # shap
        c = calculate_shap()
        self.df_final, self.explainer = c.find(model, df, prediction_col, is_classification, model_name=model_name)

        # prediction col
        self.df_final[y_variable_predict] = prediction_col

        self.df_final[y_variable] = y

        # additional inputs.
        if is_classification == True:
            # find and add probabilities in the dataset.
            prediction_col_prob = model.predict_proba(df.to_numpy())
            pd_prediction_col_prob = pd.DataFrame(prediction_col_prob)

            for c in pd_prediction_col_prob.columns:
                self.df_final["Probability_" + str(c)] = list(pd_prediction_col_prob[c])

            classes = []
            for c in pd_prediction_col_prob.columns:
                classes.append(str(c))
            self.param["classes"] = classes

            try:
                expected_values_by_class = self.explainer.expected_value
            except:
                expected_values_by_class = []
                for c in range(len(classes)):
                    expected_values_by_class.append(1 / len(classes))

            self.param["expected_values"] = expected_values_by_class
        else:
            try:
                expected_values = self.explainer.expected_value
                self.param["expected_values"] = [expected_values]
            except:
                expected_value = [round(np.array(y).mean(), 2)]
                self.param["expected_values"] = expected_value

        self.param["is_classification"] = is_classification
        self.param["model_name"] = model_name
        self.param["model"] = model
        self.param["columns"] = df.columns
        self.param["y_variable"] = y_variable
        self.param["y_variable_predict"] = y_variable_predict

        # manually test all the graphs to see if all work

        g = plotly_graphs()

        __, df2 = g.feature_importance(self.df_final)
        fim, df2 = g.feature_impact(self.df_final)
        sp = g.summary_plot(self.df_final)

        return True

    def dataset_boston(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.boston()
        return X, y

    def dataset_iris(self):
        # load JS visualization code to notebook
        shap.initjs()
        X, y = shap.datasets.iris()
        return X, y

    def dataset_heloc(self):
        dataset = pd.read_csv(path_dataset + "/heloc_dataset.csv")

        map_riskperformance = {"RiskPerformance": {"Good": 1, "Bad": 0}}
        dataset.replace(map_riskperformance, inplace=True)
        y = list(dataset["RiskPerformance"])
        X = dataset.drop("RiskPerformance", axis=1)
        return X, y

    def run_only_first_time(self):

        if platform == "linux" or platform == "linux2":
            try:
                run_command("curl -sL https://rpm.nodesource.com/setup_10.x | sudo bash -")
                run_command("sudo apt install nodejs")
                run_command("sudo apt install npm")
            except:
                run_command("sudo yum install nodejs")
                run_command("sudo yum install npm")
            run_command("npm install -g localtunnel")

        elif platform == "darwin":
            run_command("xcode-select --install")
            run_command("brew install nodejs")
            run_command("npm install -g localtunnel")

        elif platform == "win32":
            print("Please install nodejs, npm, and localtunnel manually")
            run_command("npm install -g localtunnel")
        elif platform == "win64":
            print("Please install nodejs, npm, and localtunnel manually")
            run_command("npm install -g localtunnel")


explainx = explain()


def run_command(command):
    # subdomain= 'explainx-'+ get_random_string(10)
    command_arr = command.split(" ")

    task = subprocess.Popen(command_arr,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    for line in iter(task.stdout.readline, b''):
        print('{0}'.format(line.decode('utf-8')), end='')
