from imports import *

"""
This class generates insights for each of the graphs. 
Do not change anything from this class.

"""

class insight_classification():
    def __init__(self):
        super(insight_classification, self).__init__()
        self.param = None

    def insight_1_feature_imp(self, df, classes=[0, 1]):

        top_important_variables = []
        average_shap_values = []
        df_numpy = df.to_numpy()
        
        for i in range(3):
            top_important_variables.append(df_numpy[-1 * (i+1)][0])
            average_shap_values.append(df_numpy[-1 * (i+1)][1])

       
        sentences = []
        sentences.append("The top three features influencing the models predicted outcome are: {}, {}. {}".format(str(top_important_variables[0]), str(top_important_variables[1]), str(top_important_variables[2])))

        # sentences.append("According to your model,  " + str(top_important_variables[0]) + " , " +
        #                  str(top_important_variables[1]) + " , " +
        #                  str(top_important_variables[2]) + " " + "are the top three most important variables.")

        return sentences

    def insight_2_global_feature_impact(self, df, outcome, expected_values, classes=[0, 1]):

        top_positive_variables = []
        average_shap_values_positive = []

        top_negative_variables = []
        average_shap_values_negative = []

        for i in range(1,4):
            top_positive_variables.append(df.iloc[-1 * (i+1)][0])
            average_shap_values_positive.append(df.iloc[-1 * (i+1)][1])

            top_negative_variables.append(df.iloc[i][0])
            average_shap_values_negative.append(df.iloc[i][1])

        sentences = []
        sentences.append("Your model has {:.2%} confidence that the overall outcome will be {}.".format(expected_values[outcome], outcome))
        
        sentences.append("\nOverall, on average, the top three variables influencing the model's prediction outcome are {} ({:.2%}), {} ({:.2%}), {} ({:.2%})".format(top_positive_variables[0], average_shap_values_positive[0], top_positive_variables[1], average_shap_values_positive[1], top_positive_variables[2], average_shap_values_positive[2]))

        sentences.append("\nOverall, on average, the top three variables influencing the model's prediction outcome are {} ({:.2%}), {} ({:.2%}), {} ({:.2%})".format(top_negative_variables[0], average_shap_values_negative[0], top_negative_variables[1], average_shap_values_negative[1], top_negative_variables[2], average_shap_values_negative[2]))

        # sentences.append("\nOverall, on average, the top three variables influencing the model's prediction outcome are" +
        #                   + "{}, ({})".format(top_positive_variables[0], str(round(average_shap_values_positive[0] * 100, 2))) +
        #                   "{}, ({})".format(top_positive_variables[1], str(round(average_shap_values_positive[1] * 100, 2))) +
        #                   "{}, ({})".format(top_positive_variables[2], str(round(average_shap_values_positive[2] * 100, 2))) +
        #                  + "\n")

        # sentences.append("\nOverall, on average, the top three variables influencing the model's prediction outcome are" +
        #                   + "{}, ({})".format(top_negative_variables[0], str(round(average_shap_values_negative[0] * 100, 2))) +
        #                   "{}, ({})".format(top_negative_variables[1], str(round(average_shap_values_negative[1] * 100, 2))) +
        #                   "{}, ({})".format(top_negative_variables[2], str(round(average_shap_values_negative[2] * 100, 2))) +
        #                  + "\n")

        # sentences.append("On average, the variable " +
        #                  top_negative_variables[0] + ", " +
        #                  top_negative_variables[1] + ",and " +
        #                  top_negative_variables[2] + " " +
        #                  "will change the probability for achieving outcome = " +
        #                  str(outcome) 
        #                  + " by " +
        #                  str(round(average_shap_values_negative[0] * 100, 2)) + "%," +
        #                  str(round(average_shap_values_negative[1] * 100, 2)) + "%, and" +
        #                  str(round(average_shap_values_negative[2] * 100, 2)) + "% respectively.")

        sentences.append("")

        return sentences

    def insight_3(self, df):

        top_positive_variables = []
        average_shap_values_positive = []

        df_numpy = df.to_numpy()

        for i in range(3):
            top_positive_variables.append(df_numpy[-1 * (i+1)][0])
            average_shap_values_positive.append(df_numpy[-1 * (i+1)][1])

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
            top_important_variables.append(df_numpy[-1 * (i+1)][0])
            average_shap_values.append(df_numpy[-1 * (i+1)][1])

        sentences = []

        sentences.append("The top three features influencing the models predicted outcome are: {}, {}. {}".format(str(top_important_variables[0]), top_important_variables[1], top_important_variables[2]))

        return sentences

    def insight_2_local_feature_impact(self, df, y_and_prob ):

        top_positive_variables = []
        average_shap_values_positive = []

        top_negative_variables = []
        average_shap_values_negative = []

        for i in range(1,4):
            top_positive_variables.append(df.iloc[-1 * (i)][0])
            average_shap_values_positive.append(df.iloc[-1 * (i)][1])

            top_negative_variables.append(df.iloc[i-1][0])
            average_shap_values_negative.append(df.iloc[i-1][1])

        sentences = []
        #sentences.append("Model Prediction : " + str(y_and_prob[0])+ ", Confidence Level : "+ str(y_and_prob[1]))

        sentences.append("Predicted Outcome: {}".format(y_and_prob[0]))
        sentences.append("Model Confidence Level: {:.0%}".format(y_and_prob[1]))

        sentences.append("\nThe top three variables influencing the model's prediction outcome are {} ({:.2%}), {} ({:.2%}), {} ({:.2%})".format(top_positive_variables[0], average_shap_values_positive[0], top_positive_variables[1], average_shap_values_positive[1], top_positive_variables[2], average_shap_values_positive[2]))

        sentences.append("\nThe top three variables influencing the model's prediction outcome are {} ({:.2%}), {} ({:.2%}), {} ({:.2%})".format(top_negative_variables[0], average_shap_values_negative[0], top_negative_variables[1], average_shap_values_negative[1], top_negative_variables[2], average_shap_values_negative[2]))

        sentences.append("")

        return sentences

    def insight_4_pdp(self, df):
        output = True
        return output

    def insight_5(self, x):
        output = True
        return output



