from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_cleaning import DataCleaning
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import re


class Pipeline(object):
    """This pipeline object takes in cleaned data and provides methods necessary to train and predict
    multiple types of models. The goal is to make a clean, customizable interface for comparing models"""

    def __init__(self, list_of_models):
        """Args:
                list_of_models (list): contains uninstantiated sklearn models to be tried
        """
        self.list_of_models = list_of_models
        self.trained_models = []

    def print_cv_results(self, feature_names, X_data, y_data):
        """Prints out the cross-validated estimate of test error for f1, recall and precision
        It also calls the get_variable_imp to report on the variable importance for each model"""
        for i, model in enumerate(self.trained_models):
            print "Model: ", model
            print "F1 score: ", self.f1_scores[i]
            print "recall score: ", self.recall_scores[i]
            print "precision score: ", self.precision_scores[i]

            self.get_variable_imp(model, feature_names)

    def get_variable_imp(self, model, feature_names):
        """Prints how important each feature is and creates a bar chart showing relative feature importance for each model"""
        if (str(model)).startswith('RandomForestClassifier') or (str(model)).startswith('GradientBoostingClassifier'):
            feat_imps = model.feature_importances_
            for j, importance in enumerate(feat_imps):
                print "Feature: ", feature_names[j]
                print "Feature importance: ", importance
            plot_feature_importance(feature_names, feat_imps)
        else:
            feat_imps = model.coef_.flatten()
            for j, importance in enumerate(feat_imps):
                print "Feature: ", feature_names[j]
                print "Feature importance: ", importance
            plot_feature_importance(feature_names, feat_imps)

    def fit_predict(self, x_data, y_data):
        """This method is meant to be called in main as a one-stop method for fitting
        the model and generating predictions for cross-validation test error estimation"""
        self.train(x_data, y_data)
        self.f1_scores, self.recall_scores, self.precision_scores = self.predict_and_cv_score(x_data, y_data)

    def train(self, x_data, y_data):
        """Goes through each model in self.list_of_models, finds best hyperparameters, then instantiates each model
        with its best hyperparameters. It also reports on what these hyperparameters were"""
        for model in self.list_of_models:
            if str(model()).startswith("LogisticRegression"):
                tuning_params = [{'C': [1, 10, 100, 100000]}]
            elif (str(model())).startswith('RandomForestClassifier') or (str(model)).startswith('GradientBoostingClassifier'):
                tuning_params = [{'max_depth': [2, 3, 5]}]
            grid = GridSearchCV(model(), tuning_params, cv=5, scoring='f1_macro')
            grid.fit(x_data, y_data)
            params = grid.best_params_
            trained_model = grid.best_estimator_
            self.trained_models.append(trained_model)
            p = re.compile(r"(.*)\(.*")
            model_name = re.match(p, str(trained_model)).group(1)
            print "for {} model, best parameters were: {}".format(model_name, params)

    def predict_and_cv_score(self, x_data, y_data):
        """Used by fit_predict to return model evaluation metrics through cross-validation"""
        f1_scores = []
        recall_scores = []
        precision_scores = []
        for model in self.trained_models:
            f1_scores.append(cross_val_score(model, x_data, y_data, scoring='f1').mean())
            recall_scores.append(cross_val_score(model, x_data, y_data, scoring='recall').mean())
            precision_scores.append(cross_val_score(model, x_data, y_data, scoring='f1').mean())
        return f1_scores, recall_scores, precision_scores

    def score(self, x_test, y_test):
        """This score function is meant to be used only for test data. One best hyperparameters
        are chosen through CV, use this method to get actual test error"""
        scores = []
        for model in self.trained_models:
            predictions = model.predict(x_test)
            scores.append(f1_score(y_test, predictions))
        return scores

def plot_feature_importance(feature_names, feature_importances):
    """Plots the top 10 feature importances for each model. For tree-based models, this is done
    with sklearn's feature_importances_ attribute. For logistic regression, it is the normalized
    coefficients"""
    feature_names = np.array(feature_names)
    top10_nx = np.argsort(feature_importances)[0:10]
    feat_import = feature_importances[top10_nx] # now sorted
    feat_import = feat_import / feat_import.max()
    feature_names = feature_names[top10_nx]
    fig = plt.figure()
    x_ind = np.arange(10)
    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, feature_names[x_ind])
    plt.show()

def roc_curve(probabilities, labels, model_name):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    sorted_probs_i = np.argsort(probabilities)

    TPRS = []
    FPRS = []
    positive_cases = sum(labels)
    neg_cases = len(labels) - positive_cases
    i = 0
    for instance in probabilities:
        #print "iteration: ", i
        predictions = (probabilities > instance)*1

        correct_positive = np.sum(predictions*labels)

        TPR = correct_positive/float(positive_cases)
        TPRS.append(TPR)

        incorrect_pos = np.sum(predictions) - correct_positive

        FPR = incorrect_pos/float(neg_cases)
        FPRS.append(FPR)
    TPRS = np.array(TPRS)
    FPRS = np.array(FPRS)

    tpr = TPRS[sorted_probs_i]
    fpr = FPRS[sorted_probs_i]

    plt.plot(fpr, tpr, label=model_name)

def plot_rocs(pipes, datasets):
    """Plot roc curves for all fitted models in multiple pipelines together on the same graph to compare"""
    for pipe_set in zip(pipes, datasets):
        pipe = pipe_set[0]
        X = pipe_set[1][0]
        y = pipe_set[1][1]
        for model in pipe.trained_models:
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model.fit(X_train, y_train)
            predicted_probs = model.predict_proba(X_test)[:,1]
            p = re.compile(r"(.*)\(.*")
            model_name = re.match(p, str(model)).group(1)
            roc_curve(predicted_probs, y_test, model_name)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot")
    plt.legend(loc='lower right')
    plt.show()


def main():
    train_path = "../data/churn_train.csv"
    test_path = "../data/churn_test.csv"
    dc_train = DataCleaning(train_path)
    dc_test = DataCleaning(test_path)
    X_train, y_train = dc_train.clean()
    X_test, y_test = dc_test.clean()

    dc_train_reg = DataCleaning(train_path)
    dc_test_reg = DataCleaning(test_path)
    X_train_reg, y_train_reg = dc_train_reg.clean(regression=True)
    X_test_reg, y_test_reg = dc_test_reg.clean(regression=True)

    train_col_names = dc_train.get_column_names()
    train_col_names_reg = dc_train_reg.get_column_names()

    rf = RandomForestClassifier
    gb = GradientBoostingClassifier
    logr = LogisticRegression

    pipe = Pipeline([rf, gb])
    pipe.fit_predict(X_train, y_train)
    pipe.print_cv_results(train_col_names, X_train, y_train)

    pipe2 = Pipeline([logr])
    pipe2.fit_predict(X_train_reg, y_train_reg)
    pipe2.print_cv_results(train_col_names_reg, X_train_reg, y_train_reg)

    plot_rocs([pipe, pipe2], [[X_train, y_train], [X_train_reg, y_train_reg]])

    test_scores = pipe.score(X_test, y_test)
    #print test_scores

if __name__ == '__main__':
    main()
