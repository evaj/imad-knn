import pandas as pd
import numpy as np

import os
import warnings
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from settings import DATA_DIRECTORY

data_sets = ['wine.data', 'diabetes.data', 'seeds.data']
activations = ['relu', 'logistic', 'tanh']
hidden_layer_sizes = [(k,) for k in range(3, 30, 3)]
solvers = ['adam', 'sgd']
early_stopping = [True, False]

# max_depths = [m for m in range(2, 20, 2)]
max_samples = [20, 40, 60, 80, 100]
max_features = [2, 4, 6, 8, 10, 12, 14]
numbers_of_estimators = [5, 10, 15, 20]
boot_strap = [True, False]

result_frame = pd.DataFrame(
    columns=['Data_set', 'Boot', 'Max_sample', 'Cv', 'Max_feature', 'Number_of_est', 'Accuracy', 'F-score'])

classifiers = {
    'wine.data': SGDClassifier(loss="log", penalty='l2', max_iter=1000, tol=1e-3),

    'diabetes.data': SGDClassifier(loss="log", penalty='elasticnet', max_iter=1000, tol=1e-3),

    'seeds.data': SGDClassifier(loss="log", penalty='elasticnet', max_iter=1000, tol=1e-3)
}


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


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for data_set in data_sets:
        print(data_set)
        X, Y = load_data_set(data_set)
        X = normalize(X)
        X = X.as_matrix()

        for boot in boot_strap:
            print('\tBoot = {}'.format(boot))
            for max_sample in max_samples:
                print('\t\tMax sample = {}'.format(max_sample))
                for max_feature in max_features:
                    print('\t\t\tMax feature = {}'.format(max_feature))
                    for number_of_estimators in numbers_of_estimators:
                        for cv in range(2, 10, 2):
                            accuracies = []
                            f_scores = []
                            for train, test in StratifiedKFold(n_splits=cv, shuffle=True).split(X, Y):
                                X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train, test)
                                clf = BaggingClassifier(base_estimator=classifiers[data_set],
                                                        n_estimators=number_of_estimators,
                                                        max_samples=min(max_sample, X_train.shape[1]),
                                                        max_features=min(max_feature, X.shape[1]),
                                                        bootstrap=boot)

                                clf.fit(X_train, Y_train)
                                predictions = clf.predict(X_test)

                                accuracies.append(accuracy_score(Y_test, predictions))
                                f_scores.append(f1_score(Y_test, predictions, average="macro"))

                            result = {'Data_set': data_set,
                                      'Boot': boot,
                                      'Max_sample': max_sample,
                                      'Max_feature': max_feature,
                                      'Number_of_est': number_of_estimators,
                                      'Cv': cv,
                                      'Accuracy': np.mean(accuracies),
                                      'F-score': np.mean(f_scores)}

                            result_frame = result_frame.append(result, ignore_index=True)
                        result_frame.to_csv("ensemble_logistic_results.csv", index=False)
