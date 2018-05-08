import os
import pandas as pd

from settings import DATA_DIRECTORY


def load_data(filename, attributes=None):
    data = pd.read_csv(os.path.join(DATA_DIRECTORY, filename) + '.data')
    return data if attributes is None else data[[attributes]]

