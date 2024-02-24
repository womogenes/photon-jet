# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import numpy as np

from scipy.special import softmax
from scipy.integrate import simpson
from scipy.optimize import root
from sklearn.metrics import confusion_matrix

from tensorflow import keras

from tabulate import tabulate
import matplotlib.pyplot as plt

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR

for task_name in ["scalar1", "axion1", "axion2"]:
    print(f"Loading data for task {task_name}...")
    
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_test.pkl", "rb") as fin:
        X_test = pickle.load(fin)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_test.pkl", "rb") as fin:
        Y_test = pickle.load(fin)
    
    print(f"Loading model...")
    cnn = keras.models.load_model(f"{MODEL_DIR}/{task_name}_cnn")
    
    y_pred = cnn.predict(X_test, batch_size=100)
    os.makedirs(f"{OUTPUT_DIR}/model_outputs", exist_ok=True)
    np.savez(f"{OUTPUT_DIR}/model_outputs/{task_name}_cnn_test.npz", raw_outputs=y_pred, y_true=Y_test)
