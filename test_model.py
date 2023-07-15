"""
test_model.py

Tests the PFN.
"""

print(f"Importing lots of stuff...")

import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import argparse
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Make tensorflow quieter

from tensorflow import keras
from utils import model_dir, output_dir
from data import get_data

def test_model(model, data):
    _, _, X_test, _, _, Y_test = data
    preds = model.predict(X_test, batch_size=500)

    pred_labels = np.argmax(preds, axis=1)
    test_labels = np.argmax(Y_test, axis=1)
    
    mask = (test_labels == pred_labels).astype(float)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels).astype(float)
    cm /= np.sum(cm, axis=1, keepdims=True) 
    
    return mask.mean(), cm


def plot_cm(cm, labels):
    assert(cm.shape[0] == len(labels))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    );
    disp.plot(cmap="Blues")

    
if __name__ == "__main__":
    # Task should be one of "scalar1", "axion1", and "axion2"
    # This code is the same as in train_model.py---should we modularize?
    parser = argparse.ArgumentParser(
        description="Train ParticleFlow on photon jet classification for a specific task."
    )
    parser.add_argument(
        "-t", "--task",
        choices=["scalar1", "axion1", "axion2"],
        help="Select which of three classficiation tasks to train."
    )
    args = parser.parse_args()
    
    print(f"Loading data...")
    data = get_data(args.task)
    for i, thing in enumerate(data):
        print(f"[{i}] {thing.shape}")
    
    model_path = f"{model_dir}/{args.task}_pfn"
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Test model on test set (cm stands for confusion matrix)
    accuracy, cm = test_model(model, data)
    
    print(f"Confusion matrix:")
    print(cm)
    print(f"Overall accuracy: {accuracy * 100:.5f}%")
    
    task2signature = {
        "scalar1": r"s\rightarrow\pi^0\pi^0",
        "axion1": r"a\rightarrow\gamma\gamma",
        "axion2": r"a\rightarrow 3\pi^0"
    }
    plot_cm(cm, [r"\pi^0", r"\gamma", task2signature[args.task]])
    
    plt.savefig(f"{output_dir}/{args.task}_confusion_matrix")
    