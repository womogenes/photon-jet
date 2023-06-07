# Point cloud-based models for photon jet classification

Paper: https://arxiv.org/abs/2203.16703

## Particle flow network

Paper: https://arxiv.org/abs/1810.05165

Currently working on cleaning up these files so that it's simple Python files instead of Jupyter notebooks.

All relevant code is in `scalar_test/pfn_test/`.

### Requirements

Requirements:
1. A way to open Jupyter notebooks
2. [`tensorflow` library](https://www.tensorflow.org/install/pip)
3. [`energyflow` library](https://energyflow.network/installation) (Note: this dependency is not really required because the PFN model doesn't actually come from here, it's re-implemented by hand. The library simply has some useful utility functions, namely `data_split` and `to_categorical`, that are useful. Might write own implementations later.)
4. [`scikit-learn`](https://scikit-learn.org/stable)
5. `tqdm` and `matplotlib`

The code is tested on Linux.

### Data preprocessing

Currently assumes the data lives in `/usatlas/atlas01/atlasdisk/users/atlas_wifeng/photon-jet/data/processed/scalar_test`; easily editable in the notebook file under the "Grab data" heading.

Required files in this directory:

1. `pi0_40-250GeV_100k.npz`
2. `gamma_40-250GeV_100k.npz`
3. `scalar1_40-250GeV_100k.npz`

Run this file; it'll save to `all_jets_point_cloud.npz` in the directory specified above.

### Training and testing the model

Run `scalar_test/pfn_test/pfn_test.ipynb` as a notebook.

There are cells under the "Compile and train the model" heading that have learning rate and epoch # settings. My recipe for obtaining the 97% accuracy model:

1. Train with lr=2e-4 for 45 epochs (should get to >80% val accuracy)
2. Train with lr=2e-5 for 45 epochs (should get to >90% val accuracy)
3. Train with lr=2e-6 for 30 epochs

This took about two hours on an Nvidia P100 GPU.

