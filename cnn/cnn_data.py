# Export data for CNN training

import os
import numpy as np
import h5py
from h5py import File as HDF5File

import yaml
with open("../config.yaml") as fin:
    config = yaml.safe_load(fin)
    data_dir = config["data_dir"]

def load_data(task_name):
    def _load_data(particle, datafile):
        print('load_data from datafile', datafile)
        d = h5py.File(datafile, 'r')
        first = np.expand_dims((d['layer_0'])[:], -1)
        second = np.expand_dims((d['layer_1'])[:], -1)
        third = np.expand_dims((d['layer_2'])[:], -1)
        four = np.expand_dims((d['layer_3'])[:], -1)
        energy = (d['energy'])[:].reshape(-1, 1) * 1000  # convert to MeV
        sizes = [
            first.shape[1],
            first.shape[2],
            second.shape[1],
            second.shape[2],
            third.shape[1],
            third.shape[2],
            four.shape[1],
            four.shape[2],
        ]
        y = [particle] * first.shape[0]

        return (
            first,
            second,
            third,
            four,
            y,
            energy,
            sizes,
        )
    
    s = [os.path.join(data_dir, p) for p in [
        'h5/pi0_40-250GeV_100k.h5',
        'h5/gamma_40-250GeV_100k.h5',
        f"h5/{task_name}_40-250GeV_100k.h5",
    ]]
    events = [1000, 1000, 1000]
    
    (
        first,
        second,
        third,
        four,
        y,
        energy,
        sizes,
    ) = [np.concatenate(t) for t in [a for a in zip(*[_load_data(1000, file) for file in s])]]
    
    # Shuffle everything around with a given random seed
    N = first.shape[0] // 3
    
    labels = np.concatenate((np.zeros(N) + 0,
                             np.zeros(N) + 1,
                             np.zeros(N) + 2))
    rng = np.random.default_rng(0)
    perm = np.random.permutation(3 * N)
    n_train = round(0.7 * perm.shape[0])
    n_test = 3 * N - n_train
    
    # ~2 sec
    X_train = (first[perm][:n_train], second[perm][:n_train])
    Y_train = labels[perm][:n_train]

    X_test = (first[perm][n_train:], second[perm][n_train:])
    Y_test = labels[perm][n_train:]
    
    return X_train, Y_train, X_test, Y_test