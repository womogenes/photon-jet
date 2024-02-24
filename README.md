# Detecting photon-jets with deep learning

This repository contains the source code for the paper ["Detecting highly collimated photon-jets from Higgs boson exotic decays with deep learning."](https://arxiv.org/abs/2401.15690)

## Project structure

This readme walks through all steps needed to replicate the figures in the paper.

This repository contains three sub-directories for each of our three model architectures: `bdt/` for Boosted Decision Trees, `cnn/` for Convolutional Neural Networks, and `pfn/` for Particle Flow Networks. Within each directory, there are scripts to train and evaluate the corresponding model, as well as scripts to create relevant plots.

We also have cumulative plots that evaluate multiple models; the scripts to generate them are contained in `final_plots/`.

## Getting started

Our models are implemented in Python and TensorFlow/Keras. We recommend using [`conda`](https://docs.conda.io/en/latest) to install all dependencies:
```
conda install --file requirements.txt
```

### Data preparation

Two datasets are required: one contains raw ECAL images, and the other contains processed BDT variables. They are available on the CERN EOS at `/eos/user/w/wifeng/photon-jet/data`. Here is a public link: <https://cernbox.cern.ch/s/qe4mYaSEkKJwWa5>

Modify `config.yaml` to point to the correct data paths:
```yaml
# config.yaml
DATA_DIR=/data/wifeng/photon-jet/data
MODEL_DIR=/data/wifeng/photon-jet/models
OUTPUT_DIR=/data/wifeng/photon-jet/output
```
Move the data files to the `DATA_DIR` directory so that it looks like this:
```
DATA_DIR/
    h5/
        axion1_40-250GeV_100k.h5
        axion2_40-250GeV_100k.h5
        gamma_40-250GeV_100k.h5
        pi0_40-250GeV_100k.h5
        scalar1_40-250GeV_100k.h5
    bdt_vars/
        axion1_1GeV_40-250GeV_100k.root
        axion2_1GeV_40-250GeV_100k.root
        gamma_1GeV_40-250GeV_100k.root
        pi0_1GeV_40-250GeV_100k.root
        scalar1_1GeV_40-250GeV_100k.root
```

## Training models

Trained models will be automatically saved to `MODEL_DIR/{bdt,cnn,pfn}_{scalar1,axion1,axion2}`. See [`tf.keras.Model.save`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save) for more information.

### Boosted Decision Tree

Run the following to train the BDT on a specific task (e.g. `scalar1`) and generate associated plots:
```
python bdt/train_bdt.py scalar1
```
There will be some warning messages that you can safely ignore. Once the script is finished, preprocessed data will be saved to `DATA_DIR`:
```
DATA_DIR/
    processed/
        bdt/
            scalar1_X_test.npy
            scalar1_X_train.npy
            scalar1_y_test.npy
            scalar1_y_train.npy
```
Some new files will also appear in `OUTPUT_DIR`:
```
OUTPUT_DIR/
    bdt_results/
        scalar1/
            output_importance.txt
            variable_hist_plots/
                input_first_Eratio.pdf
                input_first_dEs.pdf
                input_first_fraction_fside.pdf
                ...
```
`output_importance.txt` contains the relative importance of each input variable. The PDFs contain histograms of the BDT input variables.

The BDT model will be saved to `<OUTPUT_DIR>/scalar1_bdt.joblib`. See the [joblib](https://joblib.readthedocs.io/en/stable) documentation for more information on this file format.

### Convolutional Neural Network

#### CNN preprocessing

We preprocess CNN training data for efficiency. Run the following:
```
python cnn/cnn_data_export.py
```
This will save `.pkl` files into `DATA_DIR`:
```
DATA_DIR/
    cnn/
        axion1_X_test.pkl
        axion1_X_train.pkl
        axion1_Y_test.pkl
        axion1_Y_train.pkl
        axion2_X_test.pkl
        axion2_X_train.pkl
        axion2_Y_test.pkl
        axion2_Y_train.pkl
        scalar1_X_test.pkl
        scalar1_X_train.pkl
        scalar1_Y_test.pkl
        scalar1_Y_train.pkl
```

#### Training

To train a CNN on a given task, run the following:
```
python cnn/train_model.py scalar1
```
Replace `scalar1` with `axion1` or `axion2` to train on the other tasks. The model will be saved to `MODEL_DIR`, e.g. `<MODEL_DIR>/scalar1_cnn` for the `scalar1` CNN.

### Evaluation

To evaluate the CNN on a specific task, run the following:
```
python cnn/test_model.py scalar1
```
This will save a confusion matrix plot to `<OUTPUT_DIR>/cnn_results/scalar1_CNN_ConfusionMatrix.pdf`.

### Particle Flow Network

See <https://arxiv.org/abs/1810.05165> ("Energy Flow Networks: Deep Sets for Particle Jets") for the paper that introduced this architecture.

#### PFN preprocessing

The ECAL images must first be converted to point-clouds for PFN training. Run `preprocessing.py` to generate point-cloud scripts:
```
python pfn/preprocessing.py
```
This will create an additional sub-directory to the data directory with point-clouds for all particles:
```
DATA_DIR/
    h5/ ...
    bdt_vars/ ...
    processed/
        pfn/
            axion1_cloud.npy
            axion2_cloud.npy
            gamma_cloud.npy
            pi0_cloud.npy
            scalar1_cloud.npy
```
**Note:** this script is memory-intensive, and may crash with a "Killed." message. Pre-preprocessed files will be made available on CERNBox.

Next, run `data_tf_export.py` to split the clouds into training/test sets for each classification task:
```
python pfn/data_tf_export.py scalar1
python pfn/data_tf_export.py axion1
python pfn/data_tf_export.py axion2
```
**Note:** this script is also memory-intensive.

`DATA_DIR` should now look like this:
```
DATA_DIR/
    h5/ ...
    bdt_vars/ ...
    processed/
        pfn/
            axion1_cloud.npy
            axion2_cloud.npy
            gamma_cloud.npy
            pi0_cloud.npy
            scalar1_cloud.npy
            tf_dataset/
                scalar1_batched/
                    test/
                    train/
                axion1_batched/ ...
                axion2_batched/ ...
```

#### Training

To train a PFN on a particular task (scalar1, axion1, or axion2), run the following:
```
python pfn/train_model.py --task=scalar1
```
Change the `--task` argument to `scalar1`, `axion1`, or `axion2` depending on what task you want to train for. If you wish to continue training on an existing model, include the `--model` argument:
```
python pfn/train_model.py --task=scalar1 --model-dir=/data/wifeng/photon-jet/...
```
Once training is complete, the model will be saved to `<MODEL_DIR>/scalar1_pfn` (or `<MODEL_DIR>/axion1_pfn` or `<MODEL_DIR>/axion2_pfn`).

**Note:** training could take several hours, depending on your hardware. Pre-trained models will be made available on CERNBox.

#### Evaluation

To generate confusion matrix plots, run the following:
```
python test_model.py --task=scalar1
```
The `--task` argument should be one of `scalar1`, `axion1`, and `axion2`. This will save a PDF file of the confusion matrix in `<OUTPUT_DIR>/pfn_results/scalar1_PFN_ConfusionMatrix.pdf`.

## Plotting results

These are Figure 2, Figure 7, Figure 9, Figure 10, Table 2, and Table 3 in the paper.

### Figure 2

Run the following:
```
python final_plots/ecal_image_samples.py
```
This will generate the following PDF files in `OUTPUT_DIR`:
```
OUTPUT_DIR/
    final_plots/
        ecal_image_samples/
            axion1_layer_0_1GeV_40-250GeV_100k_noaxis.pdf
            axion1_layer_1_1GeV_40-250GeV_100k_noaxis.pdf
            axion1_layer_2_1GeV_40-250GeV_100k_noaxis.pdf
            axion1_layer_3_1GeV_40-250GeV_100k_noaxis.pdf
            gamma_layer_0_1GeV_40-250GeV_100k_noaxis.pdf
            gamma_layer_1_1GeV_40-250GeV_100k_noaxis.pdf
            gamma_layer_2_1GeV_40-250GeV_100k_noaxis.pdf
            gamma_layer_3_1GeV_40-250GeV_100k_noaxis.pdf
            pi0_layer_0_1GeV_40-250GeV_100k_noaxis.pdf
            pi0_layer_1_1GeV_40-250GeV_100k_noaxis.pdf
            pi0_layer_2_1GeV_40-250GeV_100k_noaxis.pdf
            pi0_layer_3_1GeV_40-250GeV_100k_noaxis.pdf
```
These plots are layer 1 from the ECAL images of sample `axion1`, `gamma`, and `pi0` particle jets.

### Figure 7 and Table 2

Once all architectures have been trained on all tasks, you can generate the receiver operating characteristic (ROC) curves and working point efficiency tables.

First, extract raw model outputs with these commands:
```
python bdt/bdt_output_export.py
python cnn/cnn_output_export.py
python pfn/pfn_output_export.py
```
Then, run:
```
python final_plots/efficiency_compare.py
```
This will put plots in `OUTPUT_DIR`:
```
OUTPUT_DIR/
    final_plots/
        ROC_curves/
            axion1_ROC_curves.pdf
            axion2_ROC_curves.pdf
            scalar1_ROC_curves.pdf
```
It will also print out a LaTeX-formatted Table 2.

### Table 3

These plots attempt to interpret some of the PFN model. First, export export 10% of its $\Sigma$ layer activations on all task with:
```
python pfn/pfn_layer_export.py scalar1 Sigma
python pfn/pfn_layer_export.py axion1 Sigma
python pfn/pfn_layer_export.py axion2 Sigma
```
This will create the following files:
```
OUTPUT_DIR/
    model_outputs/
        scalar1_Sigma_10%.npy
        axion1_Sigma_10%.npy
        axion2_Sigma_10%.npy
```
Next, run:
```
python final_plots/pfn_bdt_correlation_table.py
```
This will print out a LaTeX-formatted Table 3.

### Figure 9 and Figure 10

First, export the PFN SHAP values by running the `pfn/pfn_shap_values.ipynb` Jupyter Notebook. (For some reason using it as a Python script fails when `import shap` tries to run.) This will create `OUTPUT_DIR/pfn_results/<task_name>/PFN_SHAP_values.npy`.

Assuming you have run the 10% $\Sigma$ export from the previous section, you can now generate figures 9 and 10 with:
```
python final_plots/pfn_bdt_correlation_plots.py
```
This will generate two files:
```
OUTPUT_DIR/
    final_plots/
        PFN_SHAP_vs_BDT_axion2.pdf
        PFN_feature_correlations.png
```