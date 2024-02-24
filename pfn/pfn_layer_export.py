# Export the first few layers of a model

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR

print(f"Arguments: {sys.argv}")
if len(sys.argv) < 3:
    print(f"Usage: python model_export.py <model_name, e.g. axion2> <cut_layer, e.g. Sigma>")
    exit()
task_name = sys.argv[1]
cut_layers = [sys.argv[2]]

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Importing tensorflow...")
import tensorflow as tf

print("Importing other things...")
import numpy as np

## LOAD DATA
particles = ["pi0", "gamma", task_name]
clouds = []
for particle in particles:
    print(f"Loading 10% of data for particle {particle}...")
    clouds.append(np.load(f"{DATA_DIR}/processed/pfn/{particle}_cloud.npy")[::10])
clouds = np.vstack(clouds)
print(f"clouds.shape: {clouds.shape}")

## LOAD MODEL AND SPLIT IT
print("Loading model...")
full_model = tf.keras.models.load_model(f"{MODEL_DIR}/{task_name}_pfn")

for layer in cut_layers:
    print(f"Cutting at layer {layer} and computing its outputs...")
    tf.keras.backend.clear_session()
    model = tf.keras.models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(layer).output
    )
    outputs = model.predict(clouds, batch_size=1000)
    print(f"  {layer} outputs have shape {outputs.shape}.")
    
    save_dir = f"{OUTPUT_DIR}/model_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{task_name}_pfn_{layer}_10%.npy"
    print(f"  Saving to {save_path}...")
    np.save(save_path, outputs)
    
    print()
    

"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input (InputLayer)          [(None, 960, 4)]          0         
 masking (Masking)           (None, 960, 4)            0         
 Phi_0 (TimeDistributed)     (None, 960, 256)          1280      
 Phi_1 (TimeDistributed)     (None, 960, 256)          65792     
 Phi_2 (TimeDistributed)     (None, 960, 256)          65792     
 Phi_3 (TimeDistributed)     (None, 960, 256)          65792     
 Phi_4 (TimeDistributed)     (None, 960, 128)          32896     
 Phi_5 (TimeDistributed)     (None, 960, 128)          16512     
 Phi_6 (TimeDistributed)     (None, 960, 128)          16512     
 Phi_7 (TimeDistributed)     (None, 960, 128)          16512     
 Sigma (TFOpLa               (None, 128)               0         
 F_0 (Dense)                 (None, 256)               33024     
 F_1 (Dense)                 (None, 256)               65792     
 F_2 (Dense)                 (None, 256)               65792     
 F_3 (Dense)                 (None, 256)               65792     
 F_4 (Dense)                 (None, 128)               32896     
 F_5 (Dense)                 (None, 128)               16512     
 F_6 (Dense)                 (None, 128)               16512     
 F_7 (Dense)                 (None, 128)               16512     
 output (Dense)              (None, 3)                 387       
=================================================================
Total params: 594307 (2.27 MB)
Trainable params: 594307 (2.27 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""