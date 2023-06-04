# Up to a minute to import everything
# Make tensorflow quieter

# Computing imports
print(f"Importing computing stuff...")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

print(f"Importing tensorflow... (this could take a while)")
import tensorflow as tf
from tensorflow.keras import layers

# Useful imports
print(f"Importing display stuff...")
from tqdm import tqdm
import matplotlib.pyplot as plt

# Energyflow imports
import energyflow as ef
from energyflow.utils import data_split, to_categorical