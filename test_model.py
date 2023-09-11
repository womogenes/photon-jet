"""
test_model.py

Tests the EFN.
"""

print(f"Importing lots of stuff...")

import os
import json
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


def plot_cm(cm, labels, save_path):
    matrix = np.array(cm)[::-1,::-1]
    labels = labels[::-1]

    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i,j]
            color = "white" if value >= 0.5 else "black"
            ax.text(
                x=j, y=i,
                s=f"{value:.3f}",
                va='center', ha='center',
                color=color, size='large'
            )

    ax.tick_params(top=False,
                   bottom=False,
                   left=False,
                   right=False,
                   labelleft=True,
                   labelbottom=True,
                   labeltop=False)

    # Suppress warnings (Bad practice!)
    # Something about how "FixedFormatter should only be used together with FixedLocator"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels([""] + labels)
        ax.set_yticklabels([""] + labels, rotation="vertical")
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    # https://stackoverflow.com/questions/29988241/hide-ticks-but-show-tick-labels
    plt.xlabel("predicted label") #, fontsize=18)
    plt.ylabel("true label") #, fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")

    
if __name__ == "__main__":
    # Task should be one of "scalar1", "axion1", and "axion2"
    # This code is the same as in train_model.py---should we modularize?
    parser = argparse.ArgumentParser(
        description="Train EnergyFlow on photon jet classification for a specific task."
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
    
    model_path = f"{model_dir}/{args.task}_efn"
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Test model on test set (cm stands for confusion matrix)
    accuracy, cm = test_model(model, data)
    
    print(f"Confusion matrix:")
    print(cm)
    print(f"Overall accuracy: {accuracy * 100:.5f}%")
    
    task2label = {
        "scalar1": r"$s\rightarrow\pi^0\pi^0$",
        "axion1": r"$a\rightarrow\gamma\gamma$",
        "axion2": r"$a\rightarrow3\pi^0$"
    }
    labels = [r"$\pi^0$", r"$\gamma$", task2label[args.task]]
    plot_cm(cm, labels, f"{output_dir}/{args.task}_EFN_ConfusionMatrix.pdf")
    with open(f"{output_dir}/{args.task}_EFN_ConfusionMatrix.json", "w") as fout:
        json.dump({
            "labels": labels,
            "confusion_matrix": cm.tolist()
        }, fout)
    
    
    # Make training plots
    with open(f"{output_dir}/{args.task}_train_history.json") as fin:
        train_hist = json.load(fin)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].plot(train_hist["loss"], label="training loss")
    axs[0].plot(train_hist["val_loss"], label="validation loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")
    axs[0].legend()
    
    axs[1].plot(train_hist["accuracy"], label="training accuracy")
    axs[1].plot(train_hist["val_accuracy"], label="validation accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    plt.suptitle(f"Training history for {args.task} efn")
    
    plt.savefig(f"{output_dir}/{args.task}_train_history")
