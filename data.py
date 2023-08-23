"""
data.py

Exports a function, get_data(task) which returns
        (X_train, X_val, X_test,
         Y_train, Y_val, Y_test).

Train-val-test split is 70-0-30. (Test set doubles as val set.)
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
    assert(len(X) == len(Y)), "X and Y should have same length."
    assert(val + test < len(X)), "val + test should comprise less than total."

    N = len(X)
    train = N - val - test
    
    X_train, Y_train = X[:train], Y[:train]
    X_val, Y_val = X[train:(train + val)], Y[train:(train + val)]    
    
    if test == 0:
        X_test, Y_test = X_val, Y_val
    else:
        X_test, Y_test = X[(train + val):], Y[(train + val):]
        
    return (X_train, X_val, X_test,
            Y_train, Y_val, Y_test)


def get_data(task):
    """
    Return 6-tuple of data.
    task should be one of "scalar1", "axion1", or "axion2".
    """
    cloud_paths = ["pi0_cloud.npy", "gamma_cloud.npy", f"{task}_cloud.npy"]

    X = np.concatenate([
        np.load(f"{data_dir}/processed/{path}") \
        for path in cloud_paths
    ], axis=0)
    
    N = 100000  # Size of each dataset
    assert(len(X) == 3 * N)  # Assumption about data size

    Y = to_categorical((0,) * N + (1,) * N + (2,) * N)
    
    # Scramble in the same order
    rng = np.random.default_rng(1)
    permutation = np.random.permutation(3 * N)
    X = X[permutation]
    Y = Y[permutation]
    
    n_val = round(0.3 * 3 * N)
    n_test = 0  #int(0.1 * N)
    
    return data_split(X, Y, val=n_val, test=n_test)
