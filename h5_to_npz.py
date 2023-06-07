# Convert h5 files to numpy

import os
import numpy as np
from pathlib import Path
import h5py

from tqdm import tqdm

data_dir = "/usatlas/atlas01/atlasdisk/users/atlas_wifeng/photon-jet/data/raw_files"
h5_dir = f"{data_dir}/h5"
np_dir = f"{data_dir}/npz"

os.makedirs(np_dir, exist_ok=True)


def convert(dataset):
    name = Path(dataset).stem
    file = h5py.File(f"{h5_dir}/{dataset}")
    d = {}
    for key in file.keys():
        d[key] = np.array(file[key][:])
        
    np.savez(f"{np_dir}/{name}.npz", **d)
    

for dataset in tqdm(os.listdir(h5_dir)):
    try:
        convert(dataset)
    except OSError as e:
        print(f"OSError for {dataset}")
    