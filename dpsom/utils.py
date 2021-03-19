"""
Utility functions for the DPSOM model
"""

import numpy as np
from sklearn import metrics

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


def cluster_purity(y_pred,y_true):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)


def compute_finance_labels(df, shift=1):
    """
    Computes labels of financial data.

    Args:
        df:
        shift:

    Returns: df with newly added label columns, nr. of labels

    """
    n = len(df.columns)

    df["label t"] = df["return"].apply(lambda x: 1. if x > 0 else 0.)
    df["label t+1"] = df["return"].shift(-shift).apply(lambda x: 1. if x > 0 else 0.)
    df["label volume"] = df["volume_daily_change"].apply(lambda x: 1. if x > 0 else 0.)

    # removing last shift rows as there labels can not be computed
    df = df.iloc[:-shift]

    nr_labels = len(df.columns) - n

    return df, nr_labels


def print_trainable_vars(vars):
    total_parameters = 0
    print("\n\nTrainable variables:")
    for v in vars:
        print(v)
        shape = v.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim.value
        total_parameters += var_params
    print("Number of train params: {}".format(total_parameters))


def get_gradients(vars_, loss_):
    return tf.gradients(loss_, vars_)


def find_nearest(array, value):
    array = np.reshape(array, (-1,))
    idx = (np.abs(array - value)).argmin()
    return array[idx]