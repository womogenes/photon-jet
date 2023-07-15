"""
data.py

Exports a function, get_data(task) which returns
        (X_train, X_val, X_test,
         Y_train, Y_val, Y_test).

Train-val-test split is 80-10-10.
"""

import numpy as np
from tensorflow.keras.utils import to_categorical

from utils import data_dir

def data_split(X, Y, val, test):
    """
    Split X and Y (ndarrays where leading dimension is examples)
    Returns a 6-tuple (X_train, X_val, X_test, Y_train, Y_val, Y_test).

    val and test should be proportions of the dataset
    """
    assert(val + test < 1), "val + test should be less than 1."
    assert(len(X) == len(Y)), "X and Y should have same length."

    N = len(X)
    n_val = int(N * val)
    n_test = int(N * test)
    n_train = N - n_val - n_test

    return (
        X[:n_train], X[n_train:(n_train + n_val)], X[(n_train + n_val):],
        Y[:n_train], Y[n_train:(n_train + n_val)], X[(n_train + n_val):]
    )


def get_data(task):
    """
    Return 6-tuple of data.
    task should be one of "scalar1", "axion1", or "axion2".
    """
    cloud_paths = ["pi0_cloud.npy", "gamma_cloud.npy", f"{task}_cloud.npy"]

    X = np.concatenate([
        np.load(f"{data_dir}/processed/{path}") \
        for path in cloud_paths
    ], axis=1)
    assert(N == 100000)  # Assumption about data size

    Y = to_categorical((0,) * N + (1,) * N + (2,) * N)
    
    N = len(X)
    n_val = int(0.1 * N)
    n_test = int(0.1 * N)
    
    return data_split(X, Y, val=n_val, test=n_test)
