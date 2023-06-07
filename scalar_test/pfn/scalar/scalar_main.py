# Do testing and stuff in here

print(f"Importing lots of stuff...")

import os
import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Make tensorflow quieter

from data import get_data
from model import PFN
from train import train_model

# Make tensorflow not use too much memory
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Get the data
print(f"Getting data...")
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = get_data("all_jets_point_cloud.npz")


# Create the model
print(f"Creating model...")
Phi_sizes = [128, 128, 128, 128, 64, 64, 64]
F_sizes = [128, 128, 128, 128, 64, 64, 64]

_, n_particles, n_features = X_train.shape
print(f"features: {n_features}, particles: {n_particles}")
model = PFN(
    n_features=n_features,
    n_particles=n_particles,
    n_outputs=Y_train.shape[1],
    Phi_sizes=Phi_sizes,
    F_sizes=F_sizes
)


# Train the model
def train_iteration(lr, epochs):
    fit_history = train_model(
        model=model, 
        data=(X_train, X_val, Y_train, Y_val),
        lr=lr,
        epochs=epochs
    )
    return fit_history

def save_model():
    model_dir = "/usatlas/atlas01/atlasdisk/users/atlas_wifeng/photon-jet/models/pfn"
    cur_date = dt.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f"{model_dir}/{cur_date}", exist_ok=True)
    model.save(f"{model_dir}/{cur_date}/{dt.datetime.now()}")

    
if __name__ == "__main__":
    print(f"Training model...")
    print(f"=== Training with lr=2e-4 [{dt.datetime.now()}] ===")
    train_iteration(lr=2e-4, epochs=45)

