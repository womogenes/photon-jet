"""
Export all PFN outputs to .npy files.
    Each file will contain an array of shape (90000,)
    where entries are 0, 1, or 2. (Only test jets are used.)
    2 means signal, 0 (pion) and 1 (photon) mean background.
"""

import sys
sys.path.append("../..")

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm

from utils import data_dir, model_dir

def export_outputs(task):
    print(f"=== Exporting outputs for {task} task... ===")
    
    # Load the PFN
    print("  Loading model...")
    model = keras.models.load_model(f"{model_dir}/{task}_pfn")
    print("  Loading data...")
    test_data = tf.data.Dataset.load(f"{data_dir}/processed/tf_dataset/{task}_batched/test")

    out_raw = model.predict(test_data, batch_size=128)
    y_true = np.argmax(np.vstack([y for x, y in tqdm(test_data)]), axis=1)

    save_path = f"{task}_outputs.npz"
    print(f"  Saving to {save_path}...")
    np.savez(save_path, out_raw=out_raw, y_true=y_true)
    print()

for task in ["scalar1", "axion1", "axion2"]:
    export_outputs(task)
