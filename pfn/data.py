"""
data.py

Read prepared TensorFlow datasets.
"""

import tensorflow as tf

# Allow importing from one level higher
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATA_DIR
from tqdm import tqdm

import contextlib

# See documentation for https://www.tensorflow.org/api_docs/python/tf/data/Dataset
def get_data(task, verbose=False):
    datasets = ["train", "test"]
    res = []
    
    for name in (tqdm(datasets) if verbose else datasets):
        with open(os.devnull, "w") as null_file:
            with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
                # Your code here that produces output
                print("This message will not be printed or shown.")
                res.append(tf.data.Dataset.load(f"{DATA_DIR}/processed/pfn/tf_dataset/{task}_batched/{name}"))
    
    return tuple(res)
