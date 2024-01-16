# Point cloud-based models for photon jet classification

This repository contains source code for a Particle Flow Network for discriminating the decay signatures (signals) of various new particle decay patterns (of scalars and pseudoscalars) from background signals (of photons and pions). There are three different three-class classification tasks here:
1. Discriminate between $\pi^0$, $\gamma$, and $s\rightarrow \pi^0\pi^0$
2. Discriminate between $\pi^0$, $\gamma$, and $a\rightarrow \gamma\gamma$
3. Discriminate between $\pi^0$, $\gamma$, and $a\rightarrow 3\pi^0$

For more on the ParticleFlow architecture, see the paper here: https://arxiv.org/abs/1810.05165

## tl;dr

1. Edit `config.yaml` to point to a data directory that has a sub-directory named `h5` which contains `pi0_40-250GeV_100k.h5`, etc.
2. `python preprocessing.py`
3. `python train_model.py --task=scalar1`
4. `python test_model.py --task=scalar1`
5. Outputs can be viewed in the `output_dir` specified in `config.yaml`.

## Setup

### Dependencies

This code requires Python libraries `tensorflow`, `scikit-learn`, `tqdm`, `matplotlib`, and `pyyaml`.

### Configuration

Edit `config.yaml` to point to where you are storing data and where you want models to be saved. Also, `output_dir` is where outputs (like training history stats, plots, etc.) will be placed.

```yaml
# config.yaml

data_dir: <data_dir>
model_dir: <data_dir>
output_dir: <output_dir>
```

Prepare the data directory so that it looks like this:

```
<data_dir>/
  h5/
    - pi0_40-250GeV_100k.h5
    - gamma_40-250GeV_100k.h5
    - scalar1_40-250GeV_100k.h5
```

After we're done with preprocessing, the directory will have some new files:

```
<data_dir>/
  h5/
    - pi0_40-250GeV_100k.h5
    - gamma_40-250GeV_100k.h5
    - scalar1_40-250GeV_100k.h5
  processed/
    - pi0_cloud.npy
    - gamma_cloud.npy
    - scalar1_cloud.npy
```

## Data preprocessing

Run `preprocessing.py`. This will convert the four-layer images in the `.h5` files into `.npy` files representing point clouds. Each point cloud will have shape `(100000, 960, 4)` where
- 100000 is the no. of training examples (individual clouds)
- 960 is the number of particles in each cloud
- 4 is the number of features per point (currently $\eta$, $\phi$, energy, and layer number)

Then run `data_tf_export.ipynb`, which will convert all the `.npy` files into Tensorflow datasets. These will live in `<data_dir>/processed/tf_dataset`. This makes it easier to train/test the PFNs because data comes pre-batched.

### Training the model

Run
```bash
python train_model.py --task=scalar1
```
`scalar1` can be replaced with `axion1` or `axion2` to train for the other two tasks.

After training, the model will be saved to the directory `<model_dir>/scalar1_pfn` (or `axion1_pfn`, etc.). This currently cannot be changed. Training history will also be stored in `<output_dir>/scalar1_train_history.json`.`

### Testing the model

Run
```bash
python test_model.py --task=scalar1
```
(Again, `scalar1` can be replaced.) This will print out the confusion matrix and accuracy, as well as save a plot of the confusion matrix to `<output_dir>/scalar1_confusion_matrix.png`.
