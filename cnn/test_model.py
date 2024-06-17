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

    # Create file with efficiencies
    E = np.sum([np.sum(layer / 1000, axis=(1, 2, 3)) for layer in X_test], axis=0)  # Aggregate across phi, eta axes
    
    with open(f"{OUTPUT_DIR}/cnn_results/{task_name}_eff_CNN_1GeV.txt", "w") as fout:
        fout.write("EnergyRangeLow, EnergyRangeUp, Eff, EffErrLow, EffErrUp\n")
        for e_range_low in range(40, 250, 21):
            e_range_up = e_range_low + 21
            # Get indices of jets that fall into this energy bin
            jet_idxs = np.where((e_range_low <= E) & (E < e_range_up))[0]
            
            # Calculate efficiency
            # Number of signal jets
            n = np.sum(Y_test[jet_idxs] == 2)
            # Number of signal jets identified as signal
            k = np.sum(Y_pred[jet_idxs][np.where(Y_test[jet_idxs] == 2)] == 2)
            
            fout.write(f"{e_range_low:.1f}, {e_range_up:.1f}, {k / n:.5f}, 0.0, 0.0\n")
    
    return test_accuracy, cm

evaluate_cnn(task_name)