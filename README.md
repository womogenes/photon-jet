# Point cloud-based models for photon jet classification

Paper: https://arxiv.org/abs/2203.16703

## Particle flow network

Paper: https://arxiv.org/abs/1810.05165

Currently working on cleaning up these files so that it's simple Python files instead of Jupyter notebooks.

All relevant code is in `models/pfn/'.

### Requirements

Requirements:
1. A way to open Jupyter notebooks
2. [`tensorflow` library](https://www.tensorflow.org/install/pip)
3. [`energyflow` library](https://energyflow.network/installation) (Note: this dependency is not really required because the PFN model doesn't actually come from here, it's re-implemented by hand. The library simply has some useful utility functions, namely `data_split` and `to_categorical`, that are useful. Might write own implementations later.)
4. [`scikit-learn`](https://scikit-learn.org/stable)
5. `tqdm` and `matplotlib`

The code is tested on Linux.

### Data preprocessing

All existing data files should be stored in a directory called `<data_dir>`. Data preprocessing will use the files in this directory and will also write new files to the directory. Here's what we need: within `<data_dir>`, put a subdirectory called `h5` that contains the following files:

1. `pi0_40-250GeV_100k.h5`
2. `gamma_40-250GeV_100k.h5`
3. `scalar1_40-250GeV_100k.h5`

**First**, modify `h5_to_npz.py` and replace `data_dir = /usatlas/...` with the correct directory as described above. Then, run the file to convert everthing in `<data_dir>/h5` to `.npz` files. These will now live in `<data_dir>/npz`.

**Second**, `models/pfn/scalar/scalar_preprocessing.py` handles preprocessing of the numpy data (transforms calorimetry images into point clouds). One line in this file sets the `data_dir` variable; again, modify as appropriate.

Now run `models/pfn/scalar/scalar_preprocessing.py`; it'll save to `<data_dir>/preprocessed/all_jets_point_cloud.npz`, which now contains
1. the training data `X`, of shape `(300000, 960, 4)` (300k examples, 960 points per jet, 4 features per point)
2. the training labels `y`, of shape `(300000, 3)` which are one-hot encoded vectors of the label. (`0` - pion, `1` - photon, `2` - scalar)

### Training and testing the model

Run `scalar_test/pfn/scalar/scalar_main.ipynb` as a notebook. There's a `data_dir` in here that needs to be modified as before. (I should really put this into some config file later)

The `train_iteration` function allows one to train the model within a specified learning rate for a specified number of epochs. My recipe for obtaining the 97% accuracy model:

1. Train with lr=2e-4 for 45 epochs (should get to >80% val accuracy)
2. Train with lr=2e-5 for 45 epochs (should get to >90% val accuracy)
3. Train with lr=2e-6 for 30 epochs (should get to ~97% val accuracy)

Training took about two hours on an Nvidia P100 GPU.
