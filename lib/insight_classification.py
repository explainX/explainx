from imports import *

"""
This class calculates feature importance

Input: 


"""


class insight_classification():
    def __init__(self):
        super(insight_classification, self).__init__()
        self.param = None

    def insight_1_feature_imp(self, df, classes=[0, 1]):
        #
        top_important_variables = []
        average_shap_values = []

        df_numpy = df.to_numpy()

        for i in range(3):
            top_important_variables.append(df_numpy[-1 * i][0])
            average_shap_values.append(df_numpy[-1 * i][1])

        sentences = []
        # sentences.append(
        #     "This graph helps you identify which features in your dataset have the greatest effect on the outcomes of your machine learning model.")
        sentences.append("In this graph, you can see: " + str(top_important_variables[0]) + " " +
                         str(top_important_variables[1]) + " " +
                         str(top_important_variables[
                                 2]) + " " + "are the top three most important variables according to your machine learning model.")

        sentences.append("Remember, each variable might affect the outcome differently.")

        sentences.append("For example, On average, the variable " + str(top_important_variables[0]) + " " +
                         "will change the probability of predicting any outcome by " +
                         str(round(average_shap_values[0] * 100, 2)) + "%.")

        # append all outcomes
        if len(classes) == 2:
            outcomes = str(classes[0]) + " and " + str(classes[1])
        else:
            outcomes = ""
            for c in classes[:-1]:
                outcomes = outcomes + str(c) + ", "
            outcomes = outcomes + "and " + str(classes[-1])

        sentences.append("But variable " + str(top_important_variables[0]) + " " +
                         "might affect the outcome " + outcomes +
                         " differently.")

        # sentences.append("Use our Filter Function to see how different variables affect the outcome " + str(classes[0]))

        sentences.append(
            "So let’s scroll down and explore how each variable impacts the outcome on average.")

        return sentences

    def insight_2_global_feature_impact(self, df, outcome, expected_values, classes=[0, 1]):

        top_positive_variables = []
        average_shap_values_positive = []

        top_negative_variables = []
        average_shap_values_negative = []

        #df_numpy = df.to_numpy()

        for i in range(1,4):
            top_positive_variables.append(df.iloc[-1 * i][0])
            average_shap_values_positive.append(df.iloc[-1 * i][1])

            top_negative_variables.append(df.iloc[i][0])
            average_shap_values_negative.append(df.iloc[i][1])

        sentences = []

        sentences.append("The average probability of predicting " +
                         str(outcome) + " is " + str(round(expected_values[outcome],2)) + ".\n")

        sentences.append("\nOn average, the variable " +
                         top_positive_variables[0] + ", " +
                         top_positive_variables[1] + ",and " +
                         top_positive_variables[2] + " " +
                         "will increase the probability for achieving outcome = " +
                         str(outcome) 
                         + " by " +
                         str(round(average_shap_values_positive[0] * 100, 2)) + "%," +
                         str(round(average_shap_values_positive[1] * 100, 2)) + "%, and " +
                         str(round(average_shap_values_positive[2] * 100, 2)) + "% respectively.\n")

        sentences.append("On average, the variable " +
                         top_negative_variables[0] + ", " +
                         top_negative_variables[1] + ",and " +
                         top_negative_variables[2] + " " +
                         "will change the probability for achieving outcome = " +
                         str(outcome) 
                         + " by " +
                         str(round(average_shap_values_negative[0] * 100, 2)) + "%," +
                         str(round(average_shap_values_negative[1] * 100, 2)) + "%, and" +
                         str(round(average_shap_values_negative[2] * 100, 2)) + "% respectively.")

        sentences.append("")

        return sentences

    def insight_3(self, df):

        top_positive_variables = []
        average_shap_values_positive = []

        df_numpy = df.to_numpy()

        for i in range(3):
            top_positive_variables.append(df_numpy[-1 * i][0])
            average_shap_values_positive.append(df_numpy[-1 * i][1])

        sentences = []

        sentences.append("Please explore the PDP graph to identify how different values of " +
                         top_positive_variables[0] + ", " +
                         top_positive_variables[1] + ", and " +
                         top_positive_variables[2] + " " +
                         " affect the model decision.")

        return sentences

    def insight_1_feature_imp(self, df, classes=[0, 1]):
        #
        top_important_variables = []
        average_shap_values = []

        df_numpy = df.to_numpy()

        for i in range(1,4):
            top_important_variables.append(df_numpy[-1 * i][0])
            average_shap_values.append(df_numpy[-1 * i][1])

        sentences = []
        # sentences.append(
        #     "This graph helps you identify which features in your dataset have the greatest effect on the outcomes of your machine learning model.")
        sentences.append("In this graph, you can see: " + str(top_important_variables[0]) + " " +
                         str(top_important_variables[1]) + " " +
                         str(top_important_variables[
                                 2]) + " " + "are the top three most important variables according to your machine learning model.")

        sentences.append("Remember, each variable might affect the outcome differently.")

        sentences.append("For example, On average, the variable " + str(top_important_variables[0]) + " " +
                         "will change the probability of predicting any outcome by " +
                         str(round(average_shap_values[0] * 100, 2)) + "%.")

        # append all outcomes
        if len(classes) == 2:
            outcomes = str(classes[0]) + " and " + str(classes[1])
        else:
            outcomes = ""
            for c in classes[:-1]:
                outcomes = outcomes + str(c) + ", "
            outcomes = outcomes + "and " + str(classes[-1])

        sentences.append("But variable " + str(top_important_variables[0]) + " " +
                         "might affect the outcome " + outcomes +
                         " differently.")

        # sentences.append("Use our Filter Function to see how different variables affect the outcome " + str(classes[0]))

        sentences.append(
            "So let’s scroll down and explore how each variable impacts the outcome on average.")

        return sentences

    def insight_2_local_feature_impact(self, df, outcome, expected_values, classes=[0, 1]):

        top_positive_variables = []
        average_shap_values_positive = []

        top_negative_variables = []
        average_shap_values_negative = []

        #df_numpy = df.to_numpy()

        for i in range(1,4):
            top_positive_variables.append(df.iloc[-1 * i][0])
            average_shap_values_positive.append(df.iloc[-1 * i][1])

            top_negative_variables.append(df.iloc[i][0])
            average_shap_values_negative.append(df.iloc[i][1])

        sentences = []

        sentences.append("The average probability of predicting " +
                         str(outcome) + " is " + str(round(expected_values[outcome],2)) + ".\n")

        sentences.append("\nOn average, the variable " +
                         top_positive_variables[0] + ", " +
                         top_positive_variables[1] + ",and " +
                         top_positive_variables[2] + " " +
                         "will increase the probability for achieving outcome = " +
                         str(outcome) 
                         + " by " +
                         str(round(average_shap_values_positive[0] * 100, 2)) + "%, " +
                         str(round(average_shap_values_positive[1] * 100, 2)) + "%, and " +
                         str(round(average_shap_values_positive[2] * 100, 2)) + "% respectively.\n")

        sentences.append("On average, the variable " +
                         top_negative_variables[0] + ", " +
                         top_negative_variables[1] + ",and " +
                         top_negative_variables[2] + " " +
                         "will change the probability for achieving outcome = " +
                         str(outcome) 
                         + " by " +
                         str(round(average_shap_values_negative[0] * 100, 2)) + "%, " +
                         str(round(average_shap_values_negative[1] * 100, 2)) + "%, and " +
                         str(round(average_shap_values_negative[2] * 100, 2)) + "% respectively.")

        sentences.append("")

        return sentences

    def insight_4_pdp(self, df):
        output = True
        return output

    def insight_5(self, x):
        output = True
        return output



