# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

from config import MODEL_DIR, DATA_DIR, OUTPUT_DIR

from plot_cm import plot_cm

task_name = sys.argv[1]

def evaluate_cnn(task_name):
    print(f"Loading data for task {task_name}...")
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_X_test.pkl", "rb") as fin:
        X_test = pickle.load(fin)
    with open(f"{DATA_DIR}/processed/cnn/{task_name}_Y_test.pkl", "rb") as fin:        
        Y_test = pickle.load(fin)
    print(f"Loading model...")
    cnn = keras.models.load_model(f"{MODEL_DIR}/{task_name}_cnn")
    
    Y_pred = np.argmax(cnn.predict(X_test, batch_size=900), axis=1)
    
    cm = confusion_matrix(Y_test, Y_pred).astype(float)
    cm /= np.sum(cm, axis=1, keepdims=True)
    task2label = {
        "scalar1": r"$s\rightarrow\pi^0\pi^0$",
        "axion1": r"$a\rightarrow\gamma\gamma$",
        "axion2": r"$a\rightarrow3\pi^0$"
    }
    labels = [r"$\pi^0$", r"$\gamma$", task2label[task_name]]
    
    os.makedirs(f"{OUTPUT_DIR}/cnn_results", exist_ok=True)
    plot_cm(
        cm,
        labels=labels,
        save_path=f"{OUTPUT_DIR}/cnn_results/{task_name}_CNN_ConfusionMatrix.pdf"
    )
    test_accuracy = np.mean(Y_pred == Y_test)
    print(f"Mean test accuracy for {task_name}: {test_accuracy:.5f}")
    
    return test_accuracy, cm

evaluate_cnn(task_name)