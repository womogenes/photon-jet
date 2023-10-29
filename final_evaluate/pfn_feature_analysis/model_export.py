# Export the first few layers of a model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append("../..")

print("Importing tensorflow...")
import tensorflow as tf

print("Importing other things...")
import numpy as np
from utils import data_dir, model_dir

task_name = "scalar1"
particle_name = "scalar1"

## LOAD DATA
print(f"Loading data for particle {particle_name}...")
clouds = np.load(f"{data_dir}/processed/{particle_name}_cloud.npy")

## LOAD MODEL AND SPLIT IT
print("Loading model...")
full_model = tf.keras.models.load_model(f"{model_dir}/{task_name}_pfn")

cut_layers = ["F_0"]
for layer in cut_layers:
    print(f"Cutting at layer {layer} and computing hidden units...")
    tf.keras.backend.clear_session()
    model = tf.keras.models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("F_0").input
    )
    outputs = model.predict(clouds, batch_size=1000)
    print(f"  {layer} inputs have shape {outputs.shape}.")
    os.makedirs(f"./{task_name}_{layer}_outputs", exist_ok=True)
    save_path = f"./{task_name}_{layer}_outputs/{particle_name}.npz"
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
 tf.math.reduce_sum (TFOpLa  (None, 128)               0         
 mbda)                                                           
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