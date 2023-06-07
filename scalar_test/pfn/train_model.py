"""
Utility functions for training the PFN.
"""
import tensorflow as tf

def train_model(model, data,
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
    X_train, X_val, Y_train, Y_val = data
    
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
    
    return fit_history
    