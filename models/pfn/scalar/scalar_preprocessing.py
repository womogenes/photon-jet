print(f"Importing computational stuff...")
import sys; sys.path.append("..")

import os
import time
import h5py
import numpy as np
import math
import yaml

print(f"Importing display stuff...")
from tqdm import tqdm, trange
from pprint import pprint

print(f"Importing utilities...")
from utils import convert_size, data_dir

# ~5 sec
print(f"Loading datasets... (~5 sec)")

os.chdir(f"{data_dir}/npz")
raw_pions = dict(np.load("pi0_40-250GeV_100k.npz"))
raw_photons = dict(np.load("gamma_40-250GeV_100k.npz"))
raw_scalars = dict(np.load("scalar_40-250GeV_100k.npz"))

def norm_coords(n):
    """
    Generate list of n consecutive numbers, normally distributed.
        e.g. norm_coords(4) -> [-1.34, -0.45, 0.45, 1.34]
    """
    x = np.arange(n)
    return (x - np.mean(x)) / np.std(x)

def to_cloud(arr, tag):
    """
    Turn arr of shape (samples, rows, cols) into point clouds.
    Point cloud looks like (samples, points, features).
    Features will be a 4-vector of (eta, phi, energy, tag).
        tag will probably be layer #
    
    Points may be ragged; they will be padded in that case.
    """
    n_samples, n_rows, n_cols = arr.shape
    img_shape = (n_rows, n_cols)    
    n_points = n_rows * n_cols
    
    # This shape rebroadcast can take a bit to wrap your head around
    row_coords = np.broadcast_to(norm_coords(n_rows)[:, None], img_shape)
    col_coords = np.broadcast_to(norm_coords(n_cols)[None, :], img_shape)
    
    coords = np.stack((row_coords, col_coords), axis=2).reshape((n_points, -1))
    coords = np.expand_dims(coords, axis=0)
    
    coords = np.broadcast_to(coords, (n_samples, n_points, 2))
    new_arr = np.expand_dims(np.reshape(arr, (n_samples, -1)), axis=2)
    tag_arr = np.broadcast_to([[[tag]]], (n_samples, n_points, 1))
    
    return np.concatenate((coords, new_arr, tag_arr), axis=2)


# Process all datasets
def process_dataset(dataset):
    res = []
    layers = [f"layer_{i}" for i in range(4)]
    
    for i, layer in enumerate(layers):
        print(f"    Processing {layer}...")
        res.append(to_cloud(dataset[layer], tag=i))
        
    return np.concatenate(res, axis=1)

print(f"Processing all datasets...")
raw_datasets = {"pions": raw_pions, "photons": raw_photons, "scalars": raw_scalars}
processed = {}

for class_type, dataset in raw_datasets.items():
    print(f"Processing {class_type}...")
    processed[class_type] = process_dataset(raw_datasets[class_type])

# Mixing up all the jets
print(f"Mixing up all jets... (~5 sec)")
N = 100000
all_jets = np.concatenate(list(processed.values()), axis=0)
labels = np.array((0,) * N + (1,) * N + (2,) * N)

assert(len(labels) == len(all_jets))
order = np.random.permutation(len(labels))

all_jets = all_jets[order]
labels = labels[order]
print(f"All jets take up {convert_size(all_jets.nbytes)}")


# ~10 sec
print(f"Saving jets... (~10 sec)")
os.makedirs(f"{data_dir}/processed/scalar", exist_ok=True)
os.chdir(f"{data_dir}/processed/scalar")
np.savez(f"all_jets_point_cloud.npz", X=all_jets, y=labels)
