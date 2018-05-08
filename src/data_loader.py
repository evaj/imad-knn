import pandas as pd

DATA_DIR = '..\\data\\'


def load_data(filename, attributes=None):
    data = pd.read_csv(DATA_DIR + filename + '.data')
    return data if attributes is None else data[[attributes]]

