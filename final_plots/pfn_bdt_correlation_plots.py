import numpy as np
import matplotlib.pyplot as plt

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATA_DIR, OUTPUT_DIR

from tqdm import tqdm
from tabulate import tabulate

task_name = "axion2"
model_layer = "Sigma"

pfn_outputs = np.load(f"{OUTPUT_DIR}/model_outputs/{task_name}_pfn_{model_layer}_10%.npy")
pfn_outputs = pfn_outputs.T

# Stylize these labels for the sake of LaTeX
if "_" in model_layer:
    part, idx = model_layer.split("_")  # Something like ("F", "7")
    model_layer = f"{part}{idx}"
pfn_labels = [f"{model_layer}({i})" for i in range(pfn_outputs.shape[0])]

print("PFN outputs shape:", pfn_outputs.shape)

bdt_vars = []
for particle in ["pi0", "gamma", task_name]:
    bdt_var_npz = np.load(f"{DATA_DIR}/bdt_vars/{particle}_bdt_vars.npz")
    bdt_var_labels = list(bdt_var_npz.keys())
    bdt_vars.append(np.vstack([bdt_var_npz[key] for key in bdt_var_labels])[:,::10])

bdt_vars = np.hstack(bdt_vars)
print("BDT vars shape: ", bdt_vars.shape)


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

pfn_outputs, pfn_labels, _ = filter_zero_var(pfn_outputs, pfn_labels)
bdt_vars, bdt_var_labels, _ = filter_zero_var(bdt_vars, bdt_var_labels)

all_vars = np.vstack([pfn_outputs, bdt_vars])
all_labels = pfn_labels + bdt_var_labels

# Impute missing values
nan_locs = np.where(np.isnan(all_vars))
for feature, sample in zip(*nan_locs):
    all_vars[feature,sample] = np.nanmean(all_vars[feature,:])
    
assert not np.any(np.isnan(all_vars))

corr_mat = np.corrcoef(all_vars)
plt.imshow(np.abs(corr_mat))
plt.colorbar()
plt.title(f"Correlation between PFN and BDT variables");

def get_corr(var1, var2):
    return corr_mat[all_labels.index(var1),all_labels.index(var2)]
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
        
    return label

def plot_corr(var1, var2, title=True):
    idx1, idx2 = all_labels.index(var1), all_labels.index(var2)
    plt.scatter(all_vars[idx1], all_vars[idx2], s=0.1, linewidth=0)
    
    plt.xlabel(stylize(var1))
    plt.ylabel(stylize(var2))
    if title: plt.title(f"{task_name} PFN, layer {stylize(model_layer)}")

# For each BDT variable, find which PFN var it correlates most highly with
corrs = {}
for var1 in bdt_var_labels:
    idx1 = all_labels.index(var1)
    idx2 = np.argmax(corr_mat[idx1,:len(pfn_labels)])
    var2 = all_labels[idx2]
    corr = corr_mat[idx1,idx2]
    corrs[var1] = (var2, corr)

chosen = [
    "depth_weighted_total_e",
    "depth_weighted_total_e2",
    "secondlayer_x2",
    "secondlayer_y2",
    "prelayer_e",
    "total_e"
]
pairs = [(corrs[bdt_var][0], bdt_var) for bdt_var in chosen]

n_cols = 3
n_rows = -(-len(pairs) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.35)
# plt.suptitle(f"{task_name} PFN, layer {stylize(model_layer)}", fontsize="xx-large")
# plt.tight_layout()

# https://stackoverflow.com/questions/8248467/tight-layout-doesnt-take-into-account-figure-suptitle
plt.subplots_adjust(top=0.9)

for row in range(n_rows):
    for col in range(n_cols):
        i = row*n_cols + col        
        if i < len(pairs):
            pair = pairs[i]
            var1, var2 = pair
            plt.axes(axes[i//n_cols,i%n_cols])
            plt.title(f"Correlation: {get_corr(var1, var2):.3f}")
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,3))
            plot_corr(var1, var2, title=False)
        else:
            fig.delaxes(axes[row,col])

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_plots/PFN_feature_correlations.png")

# Correlate "max. BDT correlation" with "SHAP value"
def max_bdt_corr(pfn_var):
    idx1 = all_labels.index(pfn_var)
    idx2 = np.argmax(np.abs(corr_mat[idx1,len(pfn_labels):]))
    corr = corr_mat[idx1,(len(pfn_labels) + idx2)]
    return abs(corr)

shap_values = np.abs(np.load(f"{OUTPUT_DIR}/pfn_results/{task_name}_PFN_SHAP_values.npy"))
shap_values = np.mean(shap_values, axis=(0, 1))
print(shap_values.shape)

def get_SHAP_value(pfn_var):
    # VERY hard-coded
    n = pfn_var.removeprefix("Sigma(").removesuffix(")")
    return shap_values[int(n)]

max_bdt_corrs = []
mean_shap_values = []
for pfn_var in pfn_labels:
    max_bdt_corrs.append(max_bdt_corr(pfn_var))
    mean_shap_values.append(get_SHAP_value(pfn_var))

task2label = {
    "scalar1": r"$h_2\rightarrow\pi^0\pi^0$",
    "axion1": r"$a\rightarrow\gamma\gamma$",
    "axion2": r"$a\rightarrow3\pi^0$"
}
plt.rcParams.update({'font.size': 14})
plt.scatter(mean_shap_values, np.abs(max_bdt_corrs))
plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Best BDT correlation")
plt.title(f"SHAP value vs BDT correlation\nfor Sigma nodes in {task2label[task_name]} PFN");
plt.xscale("log")
# plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_plots/PFN_SHAP_vs_BDT_{task_name}.pdf")
