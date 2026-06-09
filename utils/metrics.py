import numpy as np

from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(y_true, y_pred):
    """
    Calculate Precision, Recall, and F1-score
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def point_adjust(y_true, y_pred):
    """
    Standard SMAP/MSL point adjustment: 
    if any point in an anomaly segment is detected, the whole segment is considered detected.
    """
    adjusted = y_pred.copy()
    i = 0
    while i < len(y_true):
        if y_true[i] == 1:
            j = i
            while j < len(y_true) and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                adjusted[i:j] = 1
            i = j
        else:
            i += 1
    return adjusted
