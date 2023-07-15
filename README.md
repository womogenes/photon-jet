# Point cloud-based models for photon jet classification

This repository contains source code for a ParticleFlow model the paper ["Probing highly collimated photon-jets with deep learning."](https://arxiv.org/abs/2203.16703) In short, we're trying to discriminate the decay signatures (signals) of various new particle decay patterns (of scalars and pseudoscalars) from background signals (of photons and pions). There are three different three-class classification tasks here:
1. Discriminate between $\pi^0$, $\gamma$, and $s\rightarrow \pi^0\pi^0$
2. Discriminate between $\pi^0$, $\gamma$, and $a\rightarrow \gamma\gamma$
3. Discriminate between $\pi^0$, $\gamma$, and $a\rightarrow 3\pi^0$

For more on the ParticleFlow architecture, see the paper here: https://arxiv.org/abs/1810.05165

## Setup

### Dependencies

This code requires Python libraries `tensorflow`, `scikit-learn`, `tqdm`, `matplotlib`, and `pyyaml`.

### Configuration

Edit `config.yaml` to point to where you are storing data and where you want models to be saved.

```
# config.yaml

data_dir: <data_dir>
model_dir: <data_dir>
```

Prepare the data directory so that it looks like this:

```
<data_dir>/
  h5/
    - pi0_40-250GeV_100k.h5
    - gamma_40-250GeV_100k.h5
    - scalar1_40-250GeV_100k.h5
```

After we're done with the two preprocessing steps, the directory will have some new files:

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

Run `preprocessing.py`. This will convert ==================

1. the training data `X`, of shape `(300000, 960, 4)` (300k examples, 960 points per jet, 4 features per point)
2. the training labels `y`, of shape `(300000, 3)` which are one-hot encoded vectors of the label. (`0` - pion, `1` - photon, `2` - scalar)

### Training and testing the model

Run `models/pfn/scalar/scalar_main.ipynb` as a notebook.

The `train_iteration` function allows one to train the model within a specified learning rate for a specified number of epochs. My recipe for obtaining the 97% accuracy model:

1. Train with lr=2e-4 for 45 epochs (should get to >80% val accuracy)
2. Train with lr=2e-5 for 45 epochs (should get to >90% val accuracy)
3. Train with lr=2e-6 for 30 epochs (should get to ~97% val accuracy)

Training took about two hours on an Nvidia P100 GPU.
