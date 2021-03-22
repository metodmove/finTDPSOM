"""
Utility functions for the DPSOM model
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

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
    return accuracy_score(y_pred_voted, y_true)


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


def compute_metrics(data_train, data_eval, save_dict, T=144, pred_steps=10, som_grid=[2, 2]):
    """
    Compute metrics for
    Args:
        data_train:
        data_eval:
        save_dict:
        N_companies:
        seed:
        T:
        pred_steps:
        som_grid:

    Returns:

    """
    results = {}

    # due to strange implementation of batching in T-DPSOM
    N_train = int(len(save_dict["x_rec_train"]) / T)
    N_eval = int(len(save_dict["x_rec_eval"]) / T)
    data_train = data_train[:N_train].copy()
    data_eval = data_eval[:N_eval].copy()
    nr_features = data_train.shape[-1]

    # reconstruction loss
    x_rec_train = np.reshape(save_dict["x_rec_train"], (-1, T, nr_features))
    x_rec_eval = np.reshape(save_dict["x_rec_eval"], (-1, T, nr_features))
    results["recon_MSE_train"] = round(mean_squared_error(np.reshape(data_train, -1), np.reshape(x_rec_train, -1)), 4)
    results["recon_MSE_eval"] = round(mean_squared_error(np.reshape(data_eval, -1), np.reshape(x_rec_eval, -1)), 4)
    results["recon_MSE_train_returns"] = round(
        mean_squared_error(np.reshape(data_train[:, :, 0], -1), np.reshape(x_rec_train[:, :, 0], -1)), 4)
    results["recon_MSE_eval_returns"] = round(
        mean_squared_error(np.reshape(data_eval[:, :, 0], -1), np.reshape(x_rec_eval[:, :, 0], -1)), 4)

    # pred loss
    data_train_pred = data_train[:, 1:, :]  # shift
    data_eval_pred = data_eval[:, 1:, :]  # shift
    data_eval_pred = data_eval[:, -pred_steps:, :]  # consider only last part of time-series for LSTM preds
    data_train_pred = data_train_pred[:, -pred_steps:, :]  # consider only last part of time-series for LSTM preds
    x_preds_train = np.reshape(save_dict["x_preds_train"], (-1, T, nr_features))[:, :-1, :]  # shift
    x_preds_eval = np.reshape(save_dict["x_preds_eval"], (-1, T, nr_features))[:, :-1, :]  # shift
    x_preds_train = x_preds_train[:, -pred_steps:, :]  # consider only last part of time-series for LSTM preds
    x_preds_eval = x_preds_eval[:, -pred_steps:, :]  # consider only last part of time-series for LSTM preds

    results["preds_MSE_train"] = round(
        mean_squared_error(np.reshape(data_train_pred, -1), np.reshape(x_preds_train, -1)), 4)
    results["preds_MSE_eval"] = round(mean_squared_error(np.reshape(data_eval_pred, -1), np.reshape(x_preds_eval, -1)),
                                      4)
    results["preds_MSE_train_returns"] = round(
        mean_squared_error(np.reshape(data_train_pred[:, :, 0], -1), np.reshape(x_preds_train[:, :, 0], -1)), 4)
    results["preds_MSE_eval_returns"] = round(
        mean_squared_error(np.reshape(data_eval_pred[:, :, 0], -1), np.reshape(x_preds_eval[:, :, 0], -1)), 4)

    # clustering train
    clusters_train, clusters_size_train = np.unique(save_dict["k_train"], return_counts=True)
    nr_clusters = som_grid[0] * som_grid[1]
    results["clusters_train"] = (len(clusters_train), nr_clusters)
    for i, k in enumerate(clusters_train):
        results["cluster_{}".format(k)] = clusters_size_train[i]

    # clustering eval
    # TODO

    return results