"""
preprocessing.py (v2)

Turn .h5 files into point clouds. Usage:
    python preprocessing.py

Takes no arguments. All files in <DATA_DIR>/h5 are automatically
    converted to point clouds.
"""

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from config import DATA_DIR

# Meat of the logic
def norm_coords(n):
    """
    Generate list of n consecutive numbers, normally distributed.
        e.g. norm_coords(4) -> [-1.34, -0.45, 0.45, 1.34]
    """
    x = np.arange(n)
    return (x - np.mean(x)) / np.std(x)

def to_cloud(arr, tag, threshold=1e-5):
    """
    Turn arr of shape (samples, rows, cols) into point clouds.
    Point cloud looks like (samples, points, features).
    Features will be a 4-vector of (eta, phi, energy, tag).
        tag will probably be layer #
    Points with energy < threshold will be zeroed out.
    
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
    
    full_cloud = np.concatenate((coords, new_arr, tag_arr), axis=2)
    
    # Filter out points with really low energy
    return full_cloud * (full_cloud[:,:,2] > threshold)[:,:,np.newaxis]


def process_dataset(path):
    """
    Takes in the .h5 object at path containing 100k jets of
        a certain particle, returns a 4d array using to_cloud
        to combine layers. Returns array of shape (100000, 960, 4)
    """
    print(f"Processing file [{path}]...")
    try:
        file = h5py.File(f"{DATA_DIR}/h5/{path}")
    except OSError as e:
        print(f"    Error processing .h5 file :(")
        return

    jets = {key: np.array(file[key][:]) for key in file.keys()}

    res = []
    layers = [f"layer_{i}" for i in range(4)]
    
    for i, layer in enumerate(layers):
        print(f"    Processing {layer}...")
        res.append(to_cloud(jets[layer], tag=i))
        
    cloud_arr = np.concatenate(res, axis=1)
    np.save(f"{DATA_DIR}/processed/{path.split('_')[0]}_cloud", cloud_arr)


if __name__ == "__main__":
    os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
    for path in os.listdir(f"{DATA_DIR}/h5"):
        process_dataset(path)
