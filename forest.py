import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from user_settings import DATA_DIRECTORY

data_sets = ['wine.data', 'diabetes.data', 'glass.data']
max_depth = [1, 2, 3, 5, 8]
max_features = [2, 4, 6, 8, 10]
numbers_of_estimators = [10, 20, 30, 40, 50, 60]
boot_strap = [True, False]

result_frame = pd.DataFrame(
    columns=['Data_set', 'Cv', 'Number_of_est', 'Max_depth', 'Accuracy', 'F-score', 'Max_feature'])


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
            for depth in max_depth:
                print('\t\tMax sample = {}'.format(depth))
                for max_feature in max_features:
                    print('\t\t\tMax feature = {}'.format(max_feature))
                    for number_of_estimators in numbers_of_estimators:
                        for cv in range(2, 10, 2):
                            accuracies = []
                            f_scores = []
                            for train, test in StratifiedKFold(n_splits=cv, shuffle=True).split(X, Y):
                                X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train, test)
                                clf = RandomForestClassifier(n_estimators=number_of_estimators,
                                                             max_features=min(max_feature, X.shape[1]),
                                                             max_depth=depth)

                                clf.fit(X_train, Y_train)
                                predictions = clf.predict(X_test)

                                accuracies.append(accuracy_score(Y_test, predictions))
                                f_scores.append(f1_score(Y_test, predictions, average="macro"))

                            result = {'Data_set': data_set,
                                      'Max_depth': depth,
                                      'Max_feature': max_feature,
                                      'Number_of_est': number_of_estimators,
                                      'Cv': cv,
                                      'Accuracy': np.mean(accuracies),
                                      'F-score': np.mean(f_scores)}

                            result_frame = result_frame.append(result, ignore_index=True)
                        result_frame.to_csv("random_forest_results.csv", index=False)