from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from itertools import chain, combinations
import pandas as pd
import numpy as np
import data_loader
from sklearn.model_selection import StratifiedKFold

datasets = ['wine', 'diabetes', 'seeds']
estimators = [GaussianNB(), MultinomialNB()]


def powerset(iterable, min_attr):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(min_attr, len(xs)+1))


result_frame = pd.DataFrame(columns=["Dataset", "CV", "Classifier", "Estimator",
                                     "Accuracy","F-score", "Precision", "Recall"])
for dataset in datasets:
    print('DATASET ' + dataset)
    data = data_loader.load_data(dataset)
    attr_columns = [col for col in data.columns if col != 'Type' and col != 'Id']
    train_data = data[attr_columns]
    labels = data['Type']
    if dataset == 'iris':
        labels = labels.astype('category').cat.codes
    labels = labels.as_matrix()
    # train_data = (train_data - train_data.mean()) / (train_data.max() - train_data.min())
    column_combinations = list(powerset(train_data.columns, len(train_data.columns)))
    data_length = data.shape[0]
    for estimator in estimators:
        for cv in range(2,10):
            for clf in [BaggingClassifier(base_estimator=estimator),
                        GradientBoostingClassifier(),
                        AdaBoostClassifier(base_estimator=estimator)]:
                accuracies = []
                recalls = []
                fscores = []
                precisions = []
                for train, test in StratifiedKFold(n_splits=cv, shuffle=True).split(train_data, labels):
                    sub_data = train_data[list(column_combinations[0])].as_matrix()
                    clf.fit(sub_data[train], labels[train])
                    predictions = clf.predict(sub_data[test.tolist()])
                    accuracies.append(accuracy_score(labels[test], predictions))
                    precisions.append(precision_score(labels[test], predictions, average="macro"))
                    recalls.append(recall_score(labels[test], predictions, average="macro"))
                    fscores.append(f1_score(labels[test], predictions, average="macro"))
                result = {"Dataset": dataset,
                          "CV": cv,
                          "Classifier": clf.__class__.__name__,
                          "Estimator": estimator.__class__.__name__,
                          'Accuracy': np.mean(accuracies),
                          'Recall': np.mean(recalls),
                          'F-score': np.mean(fscores),
                          'Precision': np.mean(precisions)}
                result_frame = result_frame.append(result, ignore_index=True)
        result_frame.to_csv("mlp_results.csv", index=False)
