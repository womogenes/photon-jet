"""
Various tests to test a model.
"""

from data import get_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def test_model(model, data):
    X_test, Y_test = data
    preds = model.predict(X_test, batch_size=500)

    test_labels = np.argmax(Y_test, axis=1)
    pred_labels = np.argmax(preds, axis=1)
    
    mask = (test_labels == pred_labels).astype(float)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels).astype(float)
    cm /= np.sum(cm, axis=1, keepdims=True)
    
    return mask.mean(), cm


def plot_cm(cm, labels):
    assert(cm.shape[0] == len(labels))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    );
    disp.plot(cmap="Blues")
