from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    k_arr = np.arange(1, 10, 2)  # array [1, 3, 5, 7, 9]
    accuracy = []

    for k in k_arr:
        valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
        num_correct = np.count_nonzero(valid_targets - valid_labels == 0)
        accuracy.append(num_correct / len(valid_labels))
        print(f'correct: {num_correct} total: {len(valid_labels)} accuracy: {num_correct / len(valid_labels)}')

    # Plot the classification accuracy on the validation set
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("K v.s. Classification Accuracy")
    plt.plot(k_arr, accuracy)
    plt.show()


if __name__ == "__main__":
    run_knn()
