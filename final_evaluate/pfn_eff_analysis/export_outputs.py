"""
Export all PFN outputs to .npy files.
    Each file will contain an array of shape (300000,)
    where entries are 0, 1, or 2.
    2 means signal, 0 (pion) and 1 (photon) mean background.
"""

import sys
sys.path.append("../..")

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils import data_dir, model_dir


clouds = {}
models = {}

def export_outputs(task, particle):
    if not particle in clouds:
        print(f"  Loading {particle} cloud...")
        clouds[particle] = np.load(f"{data_dir}/processed/{particle}_cloud.npy")
    if not task in models:
        print(f"  Loading {task} PFN...")
        models[task] = keras.models.load_model(f"{model_dir}/{task}_pfn")

    return models[task].predict(clouds[particle])


for task in ["scalar1", "axion1", "axion2"]:
    print(f"=== Exporting outputs for {task} task...")
    os.makedirs(f"./{task}_outputs", exist_ok=True)
    for particle in ["pi0", "gamma", task]:
        print(f"  Predicting on {particle} jets...")
        outputs = export_outputs(task, particle)
        np.save(f"./{task}_outputs/{particle}.npy", outputs)
