# Gather total model metrics like accuracy, rejection rate, etc.

import numpy as np
from tabulate import tabulate
import json
import os
import yaml

with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as fin:
    config = yaml.safe_load(fin)
    output_dir = config["output_dir"]

tasks = ["scalar1", "axion1", "axion2"]

table = [["", "efficiency", "pion rejection rate", "photon rejection rate"]]
for task in tasks:
    with open(f"{output_dir}/{task}_PFN_ConfusionMatrix.json") as fin:
        # Confuison matrix
        cm = np.array(json.load(fin)["confusion_matrix"])
    
    efficiency = cm[2,2]
    pion_rej = 1 - cm[0,2]
    photon_rej = 1 - cm[1,2]
    table.append([task, efficiency, pion_rej, photon_rej])
    

print(tabulate(table, floatfmt=".3f"))
print()
print(f"Overall signal identification efficiency:     {sum([table[i][1] for i in range(1, 4)]) * 100 / 3:.3f}%")
print(f"Overall pion rejection rate:                  {sum([table[i][2] for i in range(1, 4)]) * 100 / 3:.3f}%")
print(f"Overall photon rejection rate:                {sum([table[i][3] for i in range(1, 4)]) * 100 / 3:.3f}%")
