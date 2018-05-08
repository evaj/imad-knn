import data_loader
from itertools import chain, combinations
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

MIN_ATTRIBUTES = 3


def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(MIN_ATTRIBUTES, len(xs)+1))


datasets = ['iris', 'wine', 'diabetes', 'glass', 'seeds']
weights = ['uniform', 'distance']
metrics = ['minkowski', 'euclidean', 'manhattan']

for dataset in datasets:
    print('DATASET ' + dataset)
    data = data_loader.load_data(dataset)
    if dataset == 'iris':
        data['Type'] = data['Type'].astype('category').cat.codes
    column_combinations = list(powerset([column for column in data.columns if column != 'Type']))
    data_length = data.shape[0]
    k_values = max(1, int(0.005 * data.shape[0]))
    # noinspection PyTypeChecker
    k_values = [k_values] + [int(number) for number in range(1, int(data.shape[0]/data['Type'].max()))]
    for combination in column_combinations:
        for weight in weights:
            for metric in metrics:
                for k in k_values:
                    # skf = StratifiedKFold(n_splits=2)
                    # skf.get_n_splits(data[list(combination)], data[['Type']])
                    for train, test in StratifiedKFold(n_splits=5, shuffle=True).split(data, data['Type']):
                        clf = neighbors.KNeighborsClassifier(k, weights=weight, metric=metric)
                        print('K: ' + str(k) + " Combinations: " + str(list(combination)))
                        sub_data = data[list(combination)].as_matrix()
                        labels = data[['Type']].as_matrix()
                        clf.fit(sub_data[train], labels[train])
                        predictions = clf.predict(sub_data[test.tolist()])
