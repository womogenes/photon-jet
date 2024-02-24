# Export data for CNN training

import os
import numpy as np
import h5py
from h5py import File as HDF5File
import tensorflow as tf
import pickle

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR

def export_data(task_name):
    """
    Export CNN data with a train/test split.
    """
    def _load_data(particle, datafile):
        print('load_data from datafile', datafile)
        d = h5py.File(datafile, 'r')
        first = np.expand_dims((d['layer_0'])[:], -1)
        second = np.expand_dims((d['layer_1'])[:], -1)
        third = np.expand_dims((d['layer_2'])[:], -1)
        four = np.expand_dims((d['layer_3'])[:], -1)
        y = [particle] * first.shape[0]

        return (
            first,
            second,
            third,
            four,
            y,
        )
    
    s = [os.path.join(DATA_DIR, p) for p in [
        'h5/pi0_40-250GeV_100k.h5',
        'h5/gamma_40-250GeV_100k.h5',
        f"h5/{task_name}_40-250GeV_100k.h5",
    ]]
    
    events = [1000, 1000, 1000]

    array = [np.concatenate(t) for t in [a for a in zip(*[_load_data(i, file) for i, file in enumerate(s)])]]
    (
        first,
        second,
        third,
        four,
        y,
    ) = array
    
    # Shuffle everything around with a given random seed
    N = first.shape[0] // 3
    
    labels = y

    rng = np.random.default_rng(2)
    perm = np.random.permutation(3 * N)
    
    # Shuffle the jets and select 70% for training
    n_train = round(0.7 * perm.shape[0])
    n_test = (3 * N - n_train)
    
    first = first[perm]
    second = second[perm]
    third = third[perm]
    four = four[perm]
    labels = labels[perm]
    
    # ~2 sec
    X_train = (
        first[:n_train],
        second[:n_train],
        third[:n_train],
        four[:n_train],
    )
    Y_train = labels[:n_train]
    
    X_test = (
        first[n_train:(n_train + n_test)],
        second[n_train:(n_train + n_test)],
        third[n_train:(n_train + n_test)],
        four[n_train:(n_train + n_test)],
    )
    Y_test = labels[n_train:(n_train + n_test)]

    os.makedirs(f"{DATA_DIR}/processed/cnn", exist_ok=True)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_train.pkl", "wb") as fout:
        pickle.dump(X_train, fout)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_train.pkl", "wb") as fout:
        pickle.dump(Y_train, fout)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_test.pkl", "wb") as fout:
        pickle.dump(X_test, fout)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_test.pkl", "wb") as fout:
        pickle.dump(Y_test, fout)


if __name__ == "__main__":
    for task_name in ["scalar1", "axion1", "axion2"]:
        print(f"Exporting {task_name}...")
        export_data(task_name)
        print()
