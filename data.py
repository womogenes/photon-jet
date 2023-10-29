"""
data.py

Read prepared TensorFlow datasets.
"""

import tensorflow as tf
from utils import data_dir
from tqdm import tqdm
import sys

import contextlib
import os
import sys


def get_data(task, verbose=False):
    datasets = ["X_train", "X_test", "Y_train", "Y_test"]
    res = []
    
    for name in (tqdm(datasets) if verbose else datasets):
        with open(os.devnull, 'w') as null_file:
            with contextlib.redirect_stdout(null_file), contextlib.redirect_stderr(null_file):
                # Your code here that produces output
                print("This message will not be printed or shown.")
                res.append(tf.data.Dataset.load(f"{data_dir}/processed/tf_dataset/{task}/{name}"))
    
    return tuple(res)        
