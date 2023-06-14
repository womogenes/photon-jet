# Point cloud-based models for photon jet classification

Paper: https://arxiv.org/abs/2203.16703

## Particle flow network

Paper: https://arxiv.org/abs/1810.05165

All relevant code is in `models/pfn/'.

### Requirements

Requirements:
1. A way to open Jupyter notebooks
2. [`tensorflow` library](https://www.tensorflow.org/install/pip)
3. [`energyflow` library](https://energyflow.network/installation) (Note: this dependency is not really required because the PFN model doesn't actually come from here, it's re-implemented by hand. The library simply has some useful utility functions, namely `data_split` and `to_categorical`, that are useful. Might write own implementations later.)
4. [`scikit-learn`](https://scikit-learn.org/stable)
5. `tqdm`, `matplotlib`, and `pyyaml` (installable via pip)

The code is tested on Linux.

### Data preprocessing

Edit the paths in `/models/pfn/config.yaml` to point to the correct data and model directories. The variable `data_dir` should contain the path to a directory that **already** has the following structure:

```
h5/
  - pi0_40-250GeV_100k.h5
  - gamma_40-250GeV_100k.h5
  - scalar1_40-250GeV_100k.h5
```

After we're done with the two preprocessing steps, the folder `data_dir` should look like this:

```
h5/
  - pi0_40-250GeV_100k.h5
  - gamma_40-250GeV_100k.h5
  - scalar1_40-250GeV_100k.h5
npz/
  - pi0_40-250GeV_100k.npz
  - gamma_40-250GeV_100k.npz
  - scalar1_40-250GeV_100k.npz
processed/
  scalar/
      - all_jets_point_cloud.npz
```

**First**, run `h5_to_npz.py`. This will turn all `.h5` files in `<data_dir>/h5` into `.npz` files in `<data_dir>/npz`. (Folder will be automatically created.) There will be an `OSError` with `gamma_40-250GeV_100k_mass0p5GeV.h5`, and that's ok (the file isn't used currently).

**Second**, run `models/pfn/scalar/scalar_preprocessing.py`. This will preprocess of the numpy data (transforms calorimeter images into point clouds). It'll write a file (8.6 GB) to `<data_dir>/preprocessed/all_jets_point_cloud.npz`, which now contains
1. the training data `X`, of shape `(300000, 960, 4)` (300k examples, 960 points per jet, 4 features per point)
2. the training labels `y`, of shape `(300000, 3)` which are one-hot encoded vectors of the label. (`0` - pion, `1` - photon, `2` - scalar)

### Training and testing the model

Run `models/pfn/scalar/scalar_main.ipynb` as a notebook.

The `train_iteration` function allows one to train the model within a specified learning rate for a specified number of epochs. My recipe for obtaining the 97% accuracy model:

1. Train with lr=2e-4 for 45 epochs (should get to >80% val accuracy)
2. Train with lr=2e-5 for 45 epochs (should get to >90% val accuracy)
3. Train with lr=2e-6 for 30 epochs (should get to ~97% val accuracy)

Training took about two hours on an Nvidia P100 GPU.
