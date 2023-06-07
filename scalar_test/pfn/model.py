import tensorflow as tf
from tensorflow.keras import layers

def PFN(n_features,
        n_particles,
        n_outputs,
        Phi_sizes,
        F_sizes,
        name=None):
    
        inputs = layers.Input((n_particles, n_features), name="input")    
        masking_layer = layers.Masking(
            mask_value=0.,
            input_shape=(n_particles, n_features)
        )
        Phi_layers = [
            layers.Dense(size, activation="relu")
            for i, size in enumerate(Phi_sizes)
        ]
        F_layers = [
            layers.Dense(size, activation="relu", name=f"F_{i}")
            for i, size in enumerate(F_sizes)
        ]
        last_layer = layers.Dense(n_outputs, name="output")
        
        x = masking_layer(inputs)
        for layer in Phi_layers:
            x = layers.TimeDistributed(layer, name=f"Phi_{i}")(x)
        x = tf.math.reduce_sum(x, axis=1)
        for layer in F_layers:
            x = layer(x)
        x = last_layer(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)