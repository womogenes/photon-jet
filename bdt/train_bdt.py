# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR

import os
import random
import numpy as np
import time

import joblib

import tqdm
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
parameters = {'axes.labelsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'legend.fontsize': 12,
              'lines.linewidth' : 2,
              'lines.markersize' : 7}
plt.rcParams.update(parameters)

import seaborn as sns
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

task_name = sys.argv[1]

def findBin(bins, var, ibin):
    for i in range(len(bins)-1):
        if var >= bins[i] and var < bins[i+1]:
            ibin.append(i)

# get root files and convert them to array
branch_labels = {
                    "frac_first": "$f_{1}$",
                    "first_lateral_width_eta_w20": "$w_{s20}$",
                    "first_lateral_width_eta_w3": "$w_{s3}$",
                    "first_fraction_fside": "$f_{side}$",
                    "first_dEs": "$\Delta E_{s}$",
                    "first_Eratio": "$E_{ratio}$",
                    "second_R_eta": "$R_{\eta}$",
                    "second_R_phi": "$R_{\phi}$",
                    "second_lateral_width_eta_weta2": "$w_{\eta2}$"
                }
branch_names = list(branch_labels.keys())
pprint(branch_names)
branch_ene = """total_e""".split(",")

bg0Legend = "$\gamma$"
bg1Legend = "$\pi^0$"
sigLegend = {
    "scalar1": r"$h_2\rightarrow\pi^0\pi^0$",
    "axion1": r"$a\rightarrow\gamma\gamma$",
    "axion2": r"$a\rightarrow3\pi^0$"
}[task_name]

def as_matrix(tree, columns):
    """
    tree is an npz object containing string keys (columns) and np.array values
    """
    return np.stack([tree[col] for col in columns])

n_train = 70000
signal0_tree = np.load(f"{DATA_DIR}/bdt_vars/{task_name}_bdt_vars.npz")
signal0 = as_matrix(signal0_tree, columns=branch_names).T
signal0_ene = as_matrix(signal0_tree, columns=branch_ene).T
train_signal0 = signal0[:n_train]
test_signal0 = signal0[n_train:]
train_signal0_ene = signal0_ene[:n_train]
test_signal0_ene = signal0_ene[n_train:]

background0_tree = np.load(f"{DATA_DIR}/bdt_vars/gamma_bdt_vars.npz")
background0 = as_matrix(background0_tree, columns=branch_names).T
background0_ene = as_matrix(background0_tree, columns=branch_ene).T
train_background0 = background0[:n_train]
test_background0 = background0[n_train:]
train_background0_ene = background0_ene[:n_train]
test_background0_ene = background0_ene[n_train:]

background1_tree = np.load(f"{DATA_DIR}/bdt_vars/pi0_bdt_vars.npz")
background1 = as_matrix(background1_tree, columns=branch_names).T
background1_ene = as_matrix(background1_tree, columns=branch_ene).T
train_background1 = background1[:n_train]
test_background1 = background1[n_train:]
train_background1_ene = background1_ene[:n_train]
test_background1_ene = background1_ene[n_train:]

def plot_inputs(outdir, vars, branch_labels, sig, sig_w, bkg, bkg_w, bkg2, bkg2_w, sigLegend, bg0Legend, bg1Legend):    
    for n, var in enumerate(vars):
        _, bins = np.histogram(np.concatenate(
            (sig[:, n], bkg[:, n], bkg2[:, n])), bins=40)
        sns.distplot(sig[:, n], hist_kws={'weights': sig_w}, bins=bins, kde=False,
                     norm_hist=True, color='orange', label='{}'.format(sigLegend))
        sns.distplot(bkg[:, n], hist_kws={'weights': bkg_w}, bins=bins,
                     kde=False, norm_hist=True, color='b', label='{}'.format(bg0Legend))
        sns.distplot(bkg2[:, n], hist_kws={'weights': bkg2_w}, bins=bins,
                     kde=False, norm_hist=True, color='g', label='{}'.format(bg1Legend))
        
        plt.legend()
        if var == "first_dEs":
            plt.subplots_adjust(left=0.15)
        plt.xlabel('{}'.format(branch_labels[var]), loc='right', fontsize=38)
        plt.ylabel('Entries', fontsize=24)

        if var in ["second_lateral_width_eta_weta2", "first_dEs"]:
            plt.xticks(fontsize=17, rotation=45)
        else:
            plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        # https://stackoverflow.com/questions/42281851/how-to-add-padding-to-a-plot-in-python
        plt.tight_layout()
        
        plt.savefig(os.path.join(outdir, 'input_{}.pdf'.format(var)))
        plt.close()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    os.makedirs(f"{OUTPUT_DIR}/bdt_results/{task_name}/variable_hist_plots", exist_ok=True)
    plot_inputs(f"{OUTPUT_DIR}/bdt_results/{task_name}/variable_hist_plots", branch_names, branch_labels, train_signal0,
                None, train_background0, None, train_background1, None, sigLegend, bg0Legend, bg1Legend)

train_X_raw = np.concatenate(
    (train_signal0, train_background0, train_background1))
train_X_raw_ene = np.concatenate(
    (train_signal0_ene, train_background0_ene, train_background1_ene))
test_X_raw = np.concatenate(
    (test_signal0, test_background0, test_background1))
test_X_raw_ene = np.concatenate(
    (test_signal0_ene, test_background0_ene, test_background1_ene))

processLabels = {sigLegend: 2, bg0Legend: 0, bg1Legend: 1}
processColumns = [bg0Legend, sigLegend, bg1Legend]

sortedLabels = []
for key in processLabels:
    sortedLabels.append(processLabels[key])
sortedLabels.sort()

train_y_raw = np.concatenate((np.zeros(train_signal0.shape[0])+processLabels[sigLegend], np.zeros(
    train_background0.shape[0])+processLabels[bg0Legend], np.zeros(train_background1.shape[0])+processLabels[bg1Legend]))
test_y_raw = np.concatenate((np.zeros(test_signal0.shape[0])+processLabels[sigLegend], np.zeros(
    test_background0.shape[0])+processLabels[bg0Legend], np.zeros(test_background1.shape[0])+processLabels[bg1Legend]))

print('part2')
for key in processLabels:
    print("Length for", key, "is", len(
        test_y_raw[test_y_raw == processLabels[key]]))

X_train = train_X_raw
# https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32
print("check X_train NaN")
np.where(np.isnan(X_train))
print("check X_train Inf")
np.where(np.isinf(X_train))
# print("Replace X_train")
# X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# X_train = np.nan_to_num(X_train.astype(np.float32))

X_test = test_X_raw
X_test_ene = test_X_raw_ene
# X_test_comb = list(zip(X_test, X_test_ene))
# print("X_test_comb", X_test_comb)

y_train = train_y_raw.astype(int)
y_test = test_y_raw.astype(int)

# May as well save this data for convenience
os.makedirs(f"{DATA_DIR}/processed/bdt", exist_ok=True)
np.save(f"{DATA_DIR}/processed/bdt/{task_name}_X_train", X_train)
np.save(f"{DATA_DIR}/processed/bdt/{task_name}_y_train", y_train)
np.save(f"{DATA_DIR}/processed/bdt/{task_name}_X_test", X_test)
np.save(f"{DATA_DIR}/processed/bdt/{task_name}_y_test", y_test)

# Takes ~5 minutes on four GeForce RTX 2080 Tis
print('training')

n_boost_stages = 100

def fmt_seconds(s):
    mins = int(s / 60)
    secs = int(s % 60)
    return f"{mins:02d}:{secs:02d}"

def monitor(i, self, locals):
    global loss_func, start_time

    raw_preds = locals["raw_predictions"]
    loss = locals["loss_"](y_train, raw_preds)
    y_pred = np.argmax(raw_preds, axis=1)
    acc = np.mean(y_train == y_pred)
    ETA = (n_boost_stages - i - 1) * (time.time() - start_time) / (i + 1)
    print(f"Step {i+1:>3}/{n_boost_stages} - loss: {loss:.5f}, acc: {acc:.5f} | ETA: {fmt_seconds(ETA)}")

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
bdt = GradientBoostingClassifier(
    max_depth=5,
    min_samples_leaf=200,
    min_samples_split=10,
    n_estimators=n_boost_stages,
    learning_rate=0.5,
    warm_start=True
)

start_time = time.time()
bdt.fit(X_train, y_train, monitor=monitor)

print("Accuracy score (training): {0:.3f}".format(
    bdt.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(
    bdt.score(X_test, y_test)))


print('Saving the importance')
importances = bdt.feature_importances_
f = open(f"{OUTPUT_DIR}/bdt_results/{task_name}/output_importance.txt", 'w')
f.write("%-35s%-15s\n" % ('Variable Name', 'Output Importance'))
for i in range(len(branch_names)):
    f.write("%-35s%-15s\n" % (branch_names[i], importances[i]))
    print("%-35s%-15s\n" % (branch_names[i], importances[i]), file=f)
f.close()

# y_predicted = bdt.predict(X_train)
y_predicted = bdt.predict(X_test)

# Save the BDT
# https://scikit-learn.org/stable/model_persistence.html

os.makedirs(MODEL_DIR, exist_ok=True)
save_path = f"{MODEL_DIR}/{task_name}_bdt.joblib"
print(f"Saving {task_name} model to {save_path}...")
joblib.dump(bdt, save_path)
