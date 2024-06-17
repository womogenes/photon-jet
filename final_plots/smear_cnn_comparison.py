# Compare efficiency of:
#   1. CNN models trained on original data, evaluated on original data
#   2. CNN models trained on original data, evaluated on smeared data
#   3. CNN models trained on smeared data, evaluated on smeared data

import sys; sys.path.append("..")
import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, task2label
DATA_ROOT = os.path.dirname(OUTPUT_DIR)

outputs = {
    "output_data_smear_models_original": "Original model,\nsmeared data",
    "output_data_smear_models_smear": "Retrained model,\nsmeared data",
    "output": "Original model,\noriginal data"
}

print(DATA_ROOT)

def get_data(task, output_type):
    with open(os.path.join(
        DATA_ROOT,
        output_type,
        "cnn_results",
        f"{task}_eff_CNN_1GeV.txt"
    )) as fin:
        return [list(map(float, line.split(", "))) for line in fin.read().strip().split("\n")[1:]]

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=False)
for i, task in enumerate(task2label.keys()):
    print(i, task)
    for j, output in enumerate(outputs.items()):
        output_type, output_label = output
        data = np.array(get_data(task, output_type))
        axs[i].errorbar(
            data[:,0] + 21,
            data[:,2],
            xerr=21/2,
            yerr=0,
            color=f"C{j}",
            marker=["d", "s", "o"][j],
            label=output_label,
            linestyle="none"
        )
    axs[i].set_ylabel(f"{task2label[task]} efficiency")
    axs[i].set_xlabel("$E [GeV]$")
    if i == 0: axs[i].legend()
    plt.savefig("./image.pdf")

        
















