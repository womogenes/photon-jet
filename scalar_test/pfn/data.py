"""
data.py

Exports a function, get_data(dataset) which returns
        (X_train, X_val, X_test,
         Y_train, Y_val, Y_test).

Train-val-test split is 80-10-10.
"""

import os
import numpy as np
from utils import convert_size
from energyflow.utils import data_split, to_categorical

# ~10 sec
def get_data(dataset):
    disk_dir = "/usatlas/atlas01/atlasdisk/users/atlas_wifeng"
    data_dir = "photon-jet/data/processed"
    jets_path = f"{disk_dir}/{data_dir}/{dataset}"
    jets = np.load(jets_path)
    
    X, y = jets["X"], jets["y"]
    Y = to_categorical(y)
    
    N = len(X)
    n_val = int(0.1 * N)
    n_test = int(0.1 * N)
    
    return data_split(X, Y, val=n_val, test=n_test)
