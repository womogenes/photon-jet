"""
Export all PFN outputs to .npy files.
    Each file will contain an array of shape (90000,)
    where entries are 0, 1, or 2. (Only test jets are used.)
    2 means signal, 0 (pion) and 1 (photon) mean background.
"""

# ALlow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm

from config import DATA_DIR, MODEL_DIR

for task_name in ["scalar1", "axion1", "axion2"]:
    print(f"Exporting outputs for {task_name} PFN...")
    model = keras.models.load_model(f"{MODEL_DIR}/{task_name}_pfn")
    test_data = tf.data.Dataset.load(f"{DATA_DIR}/processed/pfn/tf_dataset/{task_name}_batched/test")
    out_raw = model.predict(test_data, batch_size=128)
    
    y_true = np.argmax(np.concatenate([y for x, y in tqdm(test_data, ncols=75)]))
    save_dir = f"{MODEL_DIR}/model_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{task_name}_pfn_test.npz"
    np.savez(save_path, raw_outputs=out_raw, y_true=y_true)
