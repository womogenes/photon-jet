# Utility function to plot a confusion matrix.

import matplotlib.pyplot as plt
import numpy as np
from utils import output_dir

def plot_cm(cm, labels, save_path):
    matrix = np.array(cm)[::-1,::-1]
    labels = labels[::-1]

    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]
            color = "white" if value >= 0.5 else "black"
            ax.text(
                x=j, y=i,
                s=f"{value:.3f}",
                va='center', ha='center',
                color=color, size='x-large'
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
        ax.set_xticklabels([""] + labels, fontdict={"fontsize": "large"})
        ax.set_yticklabels([""] + labels, rotation="vertical", fontdict={"fontsize": "large"})
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
    # We need to redo a CNN confusion matrix because of the s -> h_2 renaming
    plot_cm(
        [[0.979, 0.02, 0.001], [0.005, 0.995, 0.0], [0.006, 0.0, 0.993]],
        [r"$\pi^0$", r"$\gamma$", r"$h_2\rightarrow\pi^0\pi^0$"],
        f"{output_dir}/scalar1_CNN_ConfusionMatrix.pdf"
    )
    plot_cm(
        [[0.970, 0.013, 0.017], [0.015, 0.985, 0.000], [0.008, 0.000, 0.992]],
        [r"$\pi^0$", r"$\gamma$", r"$a\rightarrow\gamma\gamma$"],
        f"{output_dir}/axion1_CNN_ConfusionMatrix.pdf"
    )
    plot_cm(
        [[0.969, 0.031, 0.000], [0.002, 0.998, 0.000], [0.002, 0.000, 0.998]],
        [r"$\pi^0$", r"$\gamma$", r"$a\rightarrow 3\pi^0$"],
        f"{output_dir}/axion2_CNN_ConfusionMatrix.pdf"
    )

    

