# Do testing and stuff in here

print(f"Importing lots of stuff...")
import sys
sys.path.append("..")

import os
import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Make tensorflow quieter

from data import get_data
from model import PFN
from train_model import train_model
from utils import model_dir

# Get the data
print(f"Getting data...")
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = get_data("processed/scalar/all_jets_point_cloud.npz")


# Create the model
print(f"Creating model...")
Phi_sizes = (512,) * 4 + (256,) * 3
F_sizes = (256,) * 4 + (128,) * 3

_, n_particles, n_features = X_train.shape
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

def save_model(name):
    cur_date = dt.datetime.now().strftime("%Y-%m-%d")
    model.save(f"{model_dir}/{name}_{cur_date}")

    
if __name__ == "__main__":
    print(f"Training model...")
    print(f"=== Training with lr=2e-4 [{dt.datetime.now()}] ===")
    train_iteration(lr=2e-5, epochs=30)

