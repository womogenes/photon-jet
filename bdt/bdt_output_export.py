# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.special import softmax
import joblib

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR

for task_name in ["scalar1", "axion1", "axion2"]:
    print(f"Exporting outputs for {task_name} BDT...")
    
    X_train = np.load(f"{DATA_DIR}/processed/bdt/{task_name}_X_train.npy")
    y_train = np.load(f"{DATA_DIR}/processed/bdt/{task_name}_y_train.npy")
    X_test = np.load(f"{DATA_DIR}/processed/bdt/{task_name}_X_test.npy")
    y_test = np.load(f"{DATA_DIR}/processed/bdt/{task_name}_y_test.npy")
    bdt = joblib.load(f"{MODEL_DIR}/{task_name}_bdt.joblib")
    
    test_outputs = bdt.decision_function(X_test)
    
    save_dir = f"{OUTPUT_DIR}/model_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{task_name}_bdt_test.npz"
    np.savez(save_path, raw_outputs=test_outputs, y_true=y_test)
    
    print(test_outputs.shape)
    print(f"  Saved to {save_path}.")
