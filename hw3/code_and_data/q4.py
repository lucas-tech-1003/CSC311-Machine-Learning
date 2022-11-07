"""
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
"""
from scipy.special import logsumexp

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        subclass = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(subclass, axis=0)

    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        subclass = data.get_digits_by_label(train_data, train_labels, i)
        diff = subclass - means[i]
        covariances[i] = np.dot(diff.T, diff) / subclass.shape[0] + 0.01*np.identity(64)
    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """
    n = digits.shape[0]
    likelihood = np.zeros((n, 10))

    for i in range(10):
        sig = covariances[i]
        mu = means[i]
        for j in range(n):
            diff = digits[j] - mu
            exponent_part = -1 / 2 * (diff @ np.linalg.inv(sig) @ diff.T)
            likelihood[j][i] = -64 / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(np.linalg.det(sig)) + exponent_part
    return likelihood


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """
    n = digits.shape[0]
    likelihood_cond = np.zeros((n, 10))
    likelihood = generative_likelihood(digits, means, covariances)
    for j in range(n):
        likelihood_cond[j] = likelihood[j] - logsumexp(likelihood[j])
        # likelihood_cond[j] = likelihood[j] - np.logaddexp(likelihood[j])
    return likelihood_cond


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    n = digits.shape[0]
    total_prob = 0
    for j in range(n):
        label = int(labels[j])
        total_prob += cond_likelihood[j, label]
    avg = total_prob / n
    return avg


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # find average conditional log-likelihood on train and test datasets
    train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(f'The average conditional log-likelihood on training set is {train_avg}')
    print(f'The average conditional log-likelihood on test set is {test_avg}')
    # find accuracy on train and test datasets
    train_accuracy = np.mean(
        classify_data(train_data, means, covariances) == train_labels)
    test_accuracy = np.mean(
        classify_data(test_data, means, covariances) == test_labels)
    print(f'The accuracy on training set is {train_accuracy}')
    print(f'The accuracy on test set is {test_accuracy}')


if __name__ == '__main__':
    main()
