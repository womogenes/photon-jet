# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from config import DATA_DIR, OUTPUT_DIR
from collections import defaultdict

import matplotlib.pyplot as plt
from tabulate import tabulate

# Stylize labels with LaTeX
def stylize(label):
    if label.startswith("F"):
        if len(label) > 2:  # Unit
            F_layer = label[1]
            unit_idx = label[3:-1]
            return "$F^{[" + F_layer + "]}_{" + unit_idx + "}$"
        else:
            F_layer = label[1]
            return "$F^{[" + F_layer + "]}$"
        
    if label.startswith("Sigma"):
        unit_idx = label.removeprefix("Sigma(").removesuffix(")")
        return "$\Sigma_{" + unit_idx + "}$"
        
    return label.replace("_", "\\_")

task_name = "scalar1"
model_layer = "Sigma"

def get_bdt_pfn_corr_table(task_name, model_layer):
    """
    Returns a dictionary where keys are bdt variable names
        and values are a tuple (pfn_var, corr_coef)
    """
    bdt_vars = []
    for particle in ["pi0", "gamma", task_name]:
        bdt_var_npz = np.load(f"{DATA_DIR}/bdt_vars/{particle}_bdt_vars.npz")
        bdt_labels = list(bdt_var_npz.keys())
        bdt_vars.append(np.vstack([bdt_var_npz[key] for key in bdt_labels])[:,::10])
    
    bdt_vars = np.hstack(bdt_vars)
    print("BDT vars shape: ", bdt_vars.shape)
    
    pfn_vars = np.load(f"{OUTPUT_DIR}/model_outputs/{task_name}_pfn_{model_layer}_10%.npy")
    pfn_vars = pfn_vars.T
    
    # Stylize these labels for the sake of LaTeX
    if "_" in model_layer:
        part, idx = model_layer.split("_")  # Something like ("F", "7")
        model_layer = f"{part}{idx}"
    pfn_labels = [f"{model_layer}({i})" for i in range(pfn_vars.shape[0])]
    
    print("PFN outputs shape:", pfn_vars.shape)
    
    # Mask out things that are all zeros
    def filter_zero_var(arr, labels):
        """
        arr.shape == (n_features, n_samples)
        len(labels) == n_features
        
        Filter out all features having zero variance.
        """
        # Filter for zero variance
        all_zero_mask = np.nanvar(arr, axis=1) < 1e-6
        return (
            arr[~all_zero_mask],
            [labels[i] for i in np.where(~all_zero_mask)[0]],
            all_zero_mask
        )
    
    print("Processing variables...")
    pfn_vars, pfn_labels, _ = filter_zero_var(pfn_vars, pfn_labels)
    bdt_vars, bdt_labels, _ = filter_zero_var(bdt_vars, bdt_labels)
    all_vars = np.vstack([pfn_vars, bdt_vars])
    all_labels = pfn_labels + bdt_labels
    
    # Impute missing values
    nan_locs = np.where(np.isnan(all_vars))
    for feature, sample in zip(*nan_locs):
        all_vars[feature,sample] = np.nanmean(all_vars[feature,:])
        
    assert not np.any(np.isnan(all_vars))
    
    # Compute final table
    corr_mat = np.corrcoef(all_vars)
    corrs = {}
    for var1 in bdt_labels:
        idx1 = all_labels.index(var1)
        idx2 = np.argmax(corr_mat[idx1,:len(pfn_labels)])
        var2 = all_labels[idx2]
        corr = corr_mat[idx1,idx2]
        corrs[var1] = (var2, corr)
    
    return corrs

tasks = ["scalar1", "axion1", "axion2"]
corrs = {
    task_name: get_bdt_pfn_corr_table(task_name, "Sigma") \
    for task_name in tasks
}

descriptions = defaultdict(str, {
    "depth_weighted_total_e": "Summed energy across all 960 calorimeter cells, directly weighted by layer (0 for pre-sampling layer, 1 for the first, 2 for the second, 3 for the third)",
    "depth_weighted_total_e2": "Summed energy squared across all cells, directly weighted by layer",
    "total_e": "Summed energy across all 960 calorimeter cells, unweighted",
    "secondlayer_e": "Summed energy across all 256 cells in the second layer",
    "firstlayer_x2": "Summed energy across first layer, weighted by $x$-coordinate squared $(\phi^2)$",
    "firstlayer_y2": "Summed energy across first layer, weighted by $y$-coordinate squared $(\eta^2)$",
    "secondlayer_x2": "Summed energy across second layer, weighted by $x$-coordinate squared $(\phi^2)$",
    "secondlayer_y2": "Summed energy across second layer, weighted by $y$-coordinate squared $(\eta^2)$",
    "prelayer_e": "Summed energy across pre-sampling layer"
})

chosen = [
    "depth_weighted_total_e",
    "depth_weighted_total_e2",
    "secondlayer_x2",
    "secondlayer_y2",
    "prelayer_e",
    "total_e"
]

bdt_vars = chosen
table = []
avg_corrs = []

for bdt_var in bdt_vars:
    if not bdt_var in chosen:
        continue
    table.append([f"{stylize(bdt_var)}: {descriptions[bdt_var]}"])
    for task_name in tasks:
        unit, corr = corrs[task_name][bdt_var]
        table[-1].append(f"{stylize(unit)} ({corr:.3f})")
    table[-1].append(np.mean([abs(corrs[task_name][bdt_var][1]) for task_name in tasks]))

table.sort(key=lambda row: -row[-1])
table = [row[:-1] for row in table]
print(tabulate(
    table,
    headers=(["BDT variable"] + [f"{task_name} PFN" for task_name in tasks]),
    tablefmt="latex_raw"
))


