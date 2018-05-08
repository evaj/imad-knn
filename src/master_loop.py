import data_loader
from itertools import chain, combinations
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
import numpy as np

MIN_ATTRIBUTES = 3


def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(MIN_ATTRIBUTES, len(xs)+1))


datasets = ['iris', 'wine', 'diabetes', 'glass', 'seeds']
weights = ['uniform', 'distance']
metrics = ['minkowski', 'euclidean', 'manhattan']

result_frame = pd.DataFrame(columns=["K", "Metric", "Dataset", "Attributes", "Voting",
                                     "Accuracy", "F-score", "Precision", "Recall"])
for dataset in datasets:
    print('DATASET ' + dataset)
    data = data_loader.load_data(dataset)
    if dataset == 'iris':
        data['Type'] = data['Type'].astype('category').cat.codes
    column_combinations = list(powerset([column for column in data.columns if column != 'Type']))
    data_length = data.shape[0]
    # k_values = max(1, int(0.005 * data.shape[0]))
    # noinspection PyTypeChecker
    # k_values = [k_values] + [int(number) for number in range(1, int(data.shape[0]/data['Type'].max()))]
    k_values = np.arange(3, 30, 2)
    for combination in column_combinations:
        for weight in weights:
            for metric in metrics:
                for k in k_values:
                    accuracies = []
                    recalls = []
                    fscores = []
                    precisions = []
                    for train, test in StratifiedKFold(n_splits=5, shuffle=True).split(data, data['Type']):
                        clf = neighbors.KNeighborsClassifier(k, weights=weight, metric=metric)
                        print('K: ' + str(k) + " Combinations: " + str(list(combination)))
                        sub_data = data[list(combination)].as_matrix()
                        labels = data[['Type']].as_matrix()
                        clf.fit(sub_data[train], labels[train])
                        predictions = clf.predict(sub_data[test.tolist()])
                        accuracies.append(accuracy_score(labels[test], predictions))
                        precisions.append(precision_score(labels[test], predictions, average="macro"))
                        recalls.append(recall_score(labels[test], predictions, average="macro"))
                        fscores.append(f1_score(labels[test], predictions, average="macro"))
                    result = {'K': k, "Metric": metric, "Dataset": dataset, "Voting": weights,
                              "Attributes": list(combination),
                              'Accuracy': np.mean(accuracies),
                              'Recall': np.mean(recalls),
                              'F-score': np.mean(fscores),
                              'Precision': np.mean(precisions)}
                    result_frame = result_frame.append(result, ignore_index=True)
                result_frame.to_csv("results.csv", index=False)
