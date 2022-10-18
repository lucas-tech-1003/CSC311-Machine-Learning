# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N_train = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x),
                   axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N_train))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N_train = x_train.shape[0]
    # Compute the distance-based weights
    dist = l2(x_train, test_datum.T)  # a Nx1 distance vector
    denominator = np.sum(np.exp(- (dist / (2 * np.square(tau)))))
    a = (np.exp(- (dist / (2 * np.square(tau))))) / denominator
    a = np.array(a).reshape(-1, 1)
    A = np.zeros((N_train, N_train))
    for i in range(N_train):
        A[i,i] = a[i]

    # computes the optimal weight Dx1
    w = np.linalg.inv(x_train.T @ A @ x_train + lam * np.identity(d)) @ x_train.T @ A @ y_train

    # predicts y_hat
    y_hat = np.dot(test_datum.T, w)

    return y_hat


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    # Split x into training, validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=val_frac, random_state=42)
    x_train = x_train.reshape(-1,d)
    x_valid = x_valid.reshape(-1,d)
    y_train = y_train.reshape(-1,1)
    y_valid = y_valid.reshape(-1,1)
    train_loss = []
    valid_loss = []
    for tau in taus:
        total_train_loss = 0
        total_valid_loss = 0
        # Training set
        for i in range(len(x_train)):
            example = x_train[i].reshape(-1,1)
            y_hat = LRLS(example, x_train, y_train, tau)
            total_train_loss += (y_train[i] - y_hat) ** 2
        train_loss.append((total_train_loss / len(y_train)).ravel())
        # Validation set
        for i in range(len(x_valid)):
            example = x_valid[i].reshape(-1,1)
            y_hat = LRLS(example, x_train, y_train, tau)
            total_valid_loss += (y_valid[i] - y_hat) ** 2
        valid_loss.append((total_valid_loss / len(y_valid)).ravel())

    return train_loss, valid_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses, label="training")
    plt.semilogx(test_losses, label="validation")
    plt.legend()
    plt.show()
