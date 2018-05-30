import pandas as pd
import numpy as np

import os
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from settings import DATA_DIRECTORY

data_sets = ['wine.data', 'diabetes.data', 'seeds.data']
activations = ['relu', 'logistic', 'tanh']
hidden_layer_sizes = [(k, ) for k in range(3, 30, 3)]
solvers = ['adam', 'sgd']
early_stopping = [True, False]

# result_frame = pd.DataFrame(columns=['Data_set', 'Layer_size', 'Activation', 'CV', 'Stopping', 'Solver', 'Accuracy', 'F-score'])
result_frame = pd.DataFrame(columns=['Data_set', 'CV', 'Penalty', 'Accuracy', 'F-score'])


def attributes_from_data(data):
    return [attribute for attribute in data.columns if attribute != 'Type']


def load_data_set(data_set):
    data = pd.read_csv(os.path.join(DATA_DIRECTORY, data_set))
    attributes = attributes_from_data(data)

    return data[attributes], data['Type']


def normalize(data):
    return (data - data.mean()) / (data.max() - data.min())


def split_train_test(X, Y, train, test):
    return X[train], Y[train], X[test], Y[test]


for data_set in data_sets:
    print(data_set)
    X, Y = load_data_set(data_set)
    X = normalize(X)
    X = X.as_matrix()

    # for activation in activations:
    #     print('\t{}'.format(activation))
    #     for hidden_layer_size in hidden_layer_sizes:
    #         for stopping in early_stopping:
    #             for solver in solvers:
    for penalty in ['l1', 'l2', 'elasticnet']:
        for cv in range(2, 11):
            accuracies = []
            f_scores = []
            for train, test in StratifiedKFold(n_splits=cv, shuffle=True).split(X, Y):
                X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train, test)

                # clf = MLPClassifier(activation=activation,
                #                     hidden_layer_sizes=hidden_layer_size,
                #                     early_stopping=stopping,
                #                     solver=solver,
                #                     max_iter=200)
                clf = SGDClassifier(loss="log", penalty=penalty, max_iter=1000, tol=1e-3).fit(X_train, Y_train)

                clf.fit(X_train, Y_train)
                predictions = clf.predict(X_test)

                accuracies.append(accuracy_score(Y_test, predictions))
                f_scores.append(f1_score(Y_test, predictions, average="macro"))

            result = {'Data_set': data_set,
                      'CV': cv,
                      'Penalty': penalty,
                      # 'Layer_size': hidden_layer_size,
                      # 'Stopping': stopping,
                      # 'Activation': activation,
                      'Accuracy': np.mean(accuracies),
                      'F-score': np.mean(f_scores)}

            result_frame = result_frame.append(result, ignore_index=True)
        result_frame.to_csv("logistic_results.csv", index=False)
