"""
train_model.py (v2)

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

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_DIR, DATA_DIR, OUTPUT_DIR

def train_iteration(model, data,
                    lr, epochs,
                    verbose=True):
    """
    model  - the tensorflow model to train
    data   - tuple of (X_train, X_val, Y_train, Y_val)
    lr     - learning rate (float)
    epochs - number of epochs to train (int)
    
    Returns tf.keras.callbacks.History object
    """    
    print(f"\n=== Training with lr={lr} for {epochs} epochs [{dt.datetime.now()}] ===")
    
    train_data, test_data = data
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    fit_history = model.fit(x=train_data,
                            epochs=epochs,
                            validation_data=test_data,
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
    parser.add_argument(
        "-m", "--model-dir",
        help="Path of a preexiting model to train off of."
    )
    args = parser.parse_args()
    
    # Get the data
    print(f"Fetching data...")
    
    train_data = tf.data.Dataset.load(f"{DATA_DIR}/processed/tf_dataset/{args.task}_batched/train")
    test_data = tf.data.Dataset.load(f"{DATA_DIR}/processed/tf_dataset/{args.task}_batched/test")
    data = (train_data, test_data)

    #print(f"train_data size: {len(train_data)}")

    # Create the model
    if args.model_dir:
        print(f"Loading pretrained model from {args.model_dir}...")
        model = tf.keras.models.load_model(args.model_dir)
        model.summary()
        print(f"Computing metrics on test data...")
        metrics = model.evaluate(test_data)
        for metric_name, metric in zip(model.metrics_names, metrics):
            print(f"{metric_name + ':':<10} {metric}")
        print()

    else:
        print(f"Creating model...")
        Phi_sizes = (256,) * 4 + (128,) * 4
        F_sizes = (256,) * 4 + (128,) * 4

        # Extract data shape using X_train
        _, n_particles, n_features = train_data.element_spec[0].shape

        print(f"train_data spec:\n{train_data.element_spec}")
        
        model = PFN(
            n_features=n_features,
            n_particles=n_particles,
            n_outputs=train_data.element_spec[1].shape[1],  # Y_train
            Phi_sizes=Phi_sizes,
            F_sizes=F_sizes
        )    

    
    print(f"Training model...")
    hist = [
        train_iteration(model, data, lr=2e-4, epochs=15),
        train_iteration(model, data, lr=1e-4, epochs=15),
        train_iteration(model, data, lr=2e-5, epochs=15),
        train_iteration(model, data, lr=5e-6, epochs=10),
        train_iteration(model, data, lr=2e-6, epochs=10)
    ]
    
    # OPTIONAL: Save these training logs somewhere
    """
    full_hist = {}
    for key in ["loss", "val_loss", "accuracy", "val_accuracy"]:
        full_hist[key] = []
        for hist_part in hist:
            full_hist[key].extend(hist_part[key])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/{args.task}_train_history.json", "w") as fout:
        json.dump(full_hist, fout)
    """
    
    model_save_path = f"{MODEL_DIR}/{args.task}_pfn"
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
