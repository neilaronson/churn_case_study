import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_cleaning import DataCleaning
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


class Pipeline(object):

    def __init__(self, list_of_models):
        self.list_of_models = list_of_models

    def print_cv_results(self, feature_names):
        for i, model in enumerate(self.list_of_models):
            print "Model: ", model
            print "F1 score: ", self.f1_scores[i]
            print "recall score: ", self.recall_scores[i]
            print "precision score: ", self.precision_scores[i]

            self.get_variable_imp(model, feature_names)

    def get_variable_imp(self, model, feature_names):
        if (str(model)).startswith('RandomForestClassifier'):
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
        self.train(x_data, y_data)
        self.f1_scores, self.recall_scores, self.precision_scores = self.predict_and_cv_score(x_data, y_data)

    def train(self, x_data, y_data):
        for model in self.list_of_models:
            model.fit(x_data, y_data)

    def predict_and_cv_score(self, x_data, y_data):
        f1_scores = []
        recall_scores = []
        precision_scores = []
        for model in self.list_of_models:
            f1_scores.append(cross_val_score(model, x_data, y_data, scoring='f1').mean())
            recall_scores.append(cross_val_score(model, x_data, y_data, scoring='recall').mean())
            precision_scores.append(cross_val_score(model, x_data, y_data, scoring='f1').mean())
            #confusion_matrices.append(cross_val_score(model, x_data, y_data, scoring='confusion_matrix'))
        return f1_scores, recall_scores, precision_scores

    def score(self, x_test, y_test):
        scores = []
        for model in self.list_of_models:
            predictions = model.predict(x_test)
            scores.append(f1_score(y_test, predictions))
        return scores

def plot_feature_importance(feature_names, feature_importances):
    feature_names = np.array(feature_names)
    top10_nx = np.argsort(feature_importances)[0:10]
    #import ipdb; ipdb.set_trace()
    feat_import = feature_importances[top10_nx] # now sorted
    feat_import = feat_import / feat_import.max()
    feature_names = feature_names[top10_nx]
    fig = plt.figure()
    x_ind = np.arange(10)
    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, feature_names[x_ind])
    plt.show()


def main():
    #df = pd.read_csv('data/churn_train.csv')
    dc_train = DataCleaning(instance='training')
    dc_test = DataCleaning(instance='test')
    X_train, y_train = dc_train.clean()
    X_test, y_test = dc_test.clean()
    train_col_names = dc_train.get_column_names()





    rf = RandomForestClassifier(n_estimators=1000)
    logr = LogisticRegression(C=100000)
    pipe = Pipeline([rf, logr])

    pipe.fit_predict(X_train, y_train)
    test_scores = pipe.score(X_test, y_test)
    pipe.print_cv_results(train_col_names)
    print test_scores

    #import ipdb; ipdb.set_trace()




if __name__ == '__main__':
    main()







#train model

#cross validate, score

#tune

#predict
