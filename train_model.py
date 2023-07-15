"""
main.py

Trains PFN for a given classification task.
"""

print(f"Importing lots of stuff...")
import os
import datetime as dt
import argparse
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Make tensorflow quieter

import tensorflow as tf
from data import get_data
from model import PFN
from utils import model_dir, output_dir

def train_iteration(model, data,
                    lr, epochs,
                    batch_size=100,
                    verbose=True):
    """
    model  - the tensorflow model to train
    data   - tuple of (X_train, X_val, Y_train, Y_val)
    lr     - learning rate (float)
    epochs - number of epochs to train (int)
    
    Returns tf.keras.callbacks.History object
    """    
    print(f"\n=== Training with lr={lr} for {epochs} epochs [{dt.datetime.now()}] ===")
    
    X_train, X_val, _, Y_train, Y_val, _ = data
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    fit_history = model.fit(X_train, Y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, Y_val),
                            verbose=verbose)
    
    return fit_history.history


if __name__ == "__main__":
    # Task should be one of "scalar1", "axion1", and "axion2"
    parser = argparse.ArgumentParser(
        description="Train ParticleFlow on photon jet classification for a specific task."
    )
    parser.add_argument(
        "-t", "--task",
        choices=["scalar1", "axion1", "axion2"],
        help="Select which of three classficiation tasks to train."
    )
    args = parser.parse_args()
    
    # Get the data
    print(f"Fetching data...")
    data = get_data(args.task)

    # Create the model
    print(f"Creating model...")
    Phi_sizes = (256,) * 4 + (128,) * 4
    F_sizes = (256,) * 4 + (128,) * 4

    # Extract data shape using X_train
    _, n_particles, n_features = data[0].shape
    model = PFN(
        n_features=n_features,
        n_particles=n_particles,
        n_outputs=data[3].shape[1],  # Y_train
        Phi_sizes=Phi_sizes,
        F_sizes=F_sizes
    )
    
    print(f"Training model...")
    hist = [
        train_iteration(model, data, lr=2e-4, epochs=45),
        train_iteration(model, data, lr=2e-5, epochs=45),
        train_iteration(model, data, lr=2e-6, epochs=45)
    ]
    
    # Save these training logs somewhere
    full_hist = {}
    for key in ["loss", "val_loss", "accuracy", "val_accuracy"]:
        full_hist[key] = []
        for hist_part in hist:
            full_hist[key].extend(hist_part[key])
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{args.task}_train_history.json", "w") as fout:
        json.dump(full_hist, fout)
    
    model_save_path = f"{model_dir}/{args.task}_pfn"
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
