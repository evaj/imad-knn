import data_loader
from itertools import chain, combinations
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
import numpy as np

MIN_ATTRIBUTES = 3


def powerset(iterable, min_attr):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(min_attr, len(xs)+1))


def custom_function(distances):
    return list(reversed(np.argsort(distances).tolist()))


datasets = {'iris': 3, 'wine': 9, 'diabetes': 5, 'glass': 6, 'seeds': 4}
weights = ['uniform', 'distance', custom_function]
metrics = ['chebyshev', 'euclidean', 'minkowski']

result_frame = pd.DataFrame(columns=["K", "Metric", "Dataset", "Voting", "CV",
                                     "Accuracy","F-score", "Precision", "Recall"])
for dataset, min_attr in datasets.items():
    print('DATASET ' + dataset)
    data = data_loader.load_data(dataset)
    attr_columns = [col for col in data.columns if col != 'Type' and col != 'Id']
    train_data = data[attr_columns]
    labels = data['Type']

    if dataset == 'iris':
        labels = labels.astype('category').cat.codes
    labels = labels.as_matrix()
    train_data = (train_data - train_data.mean()) / (train_data.max() - train_data.min())
    column_combinations = list(powerset(train_data.columns, len(train_data.columns)))
    data_length = data.shape[0]
    k_values = np.arange(3, 30, 2)
    for combination in column_combinations:
        for weight in weights:
            for metric in metrics:
                for k in k_values:
                    for cv in range(2, 7):
                        accuracies = []
                        recalls = []
                        fscores = []
                        precisions = []
                        for train, test in StratifiedKFold(n_splits=cv, shuffle=True).split(train_data, labels):
                            clf = neighbors.KNeighborsClassifier(k, weights=weight, metric=metric)
                            print('K: ' + str(k) + " Combinations: " + str(list(combination)))
                            sub_data = train_data[list(combination)].as_matrix()
                            # labels = labels.as_matrix()
                            clf.fit(sub_data[train], labels[train])

                            predictions = clf.predict(sub_data[test.tolist()])
                            accuracies.append(accuracy_score(labels[test], predictions))
                            precisions.append(precision_score(labels[test], predictions, average="macro"))
                            recalls.append(recall_score(labels[test], predictions, average="macro"))
                            fscores.append(f1_score(labels[test], predictions, average="macro"))
                        if callable(weight):
                            weight_name = 'reverse_distance'
                        else:
                            weight_name = weight
                        result = {'K': k, "Metric": metric, "Dataset": dataset,
                                  "Voting": weight_name,
                                  "CV": cv,
                                  # "Attributes": list(combination),
                                  'Accuracy': np.mean(accuracies),
                                  'Recall': np.mean(recalls),
                                  'F-score': np.mean(fscores),
                                  'Precision': np.mean(precisions)}
                        result_frame = result_frame.append(result, ignore_index=True)
                    result_frame.to_csv("results.csv", index=False)
