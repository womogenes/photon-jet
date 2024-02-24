import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR

task_name = sys.argv[1]

def data_split(X, Y, n_test):
    """
    Split X and Y (ndarrays where leading dimension is examples)
    Returns a 6-tuple (X_train, X_test, Y_train, Y_test).
    """
    assert(len(X) == len(Y)), "X and Y should have same length."
    assert(n_test < len(X)), "n_test should comprise less than total."

    N = len(X)
    train = N - n_test
    
    X_train, Y_train = X[:train], Y[:train]
    X_test, Y_test = X[train:], Y[train:]

    return (X_train, X_test,
            Y_train, Y_test)

def get_data(task):
    """
    Return 4-tuple of data.
    task should be one of "scalar1", "axion1", or "axion2".
    """
    cloud_paths = ["pi0_cloud.npy", "gamma_cloud.npy", f"{task}_cloud.npy"]
    
    print(f"Fetching all clouds...")
    X = np.concatenate([
        np.load(f"{DATA_DIR}/processed/pfn/{path}") \
        for path in cloud_paths
    ], axis=0)
    
    N = 100000  # Size of each dataset
    assert(len(X) == 3 * N)  # Assumption about data size
    
    Y = to_categorical((0,) * N + (1,) * N + (2,) * N)
    
    # Scramble in the same order
    print(f"Scrambling order...")
    rng = np.random.default_rng(0)
    permutation = np.random.permutation(3 * N)
    X = X[permutation]
    Y = Y[permutation]
    
    n_test = round(0.3 * 3 * N)
    
    return data_split(X, Y, n_test=n_test)

# ~40 sec
(X_train, X_test, Y_train, Y_test) = get_data(task_name)

tf.executing_eagerly()
train_dataset = tf.data.Dataset.zip(
    tf.data.Dataset.from_tensor_slices(X_train),
    tf.data.Dataset.from_tensor_slices(Y_train)
).batch(64)
train_dataset.save(f"{DATA_DIR}/processed/pfn/tf_dataset/{task_name}_batched/train")
test_dataset = tf.data.Dataset.zip(
    tf.data.Dataset.from_tensor_slices(X_test),
    tf.data.Dataset.from_tensor_slices(Y_test)
).batch(64)
test_dataset.save(f"{DATA_DIR}/processed/pfn/tf_dataset/{task_name}_batched/test")
