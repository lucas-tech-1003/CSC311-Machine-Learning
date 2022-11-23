from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape


    #####################################################################
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 200
    }
    print(hyperparameters)
    weights = np.zeros(M + 1, dtype=np.float64).reshape(-1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    # Training
    entropy_train = []
    entropy_valid = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets,
                            hyperparameters)
        entropy_train.append(f)
        f_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)[0]
        entropy_valid.append(f_valid)
        weights -= hyperparameters["learning_rate"] * df
    y = logistic(weights, train_inputs, train_targets, hyperparameters)[2]
    print(f'---Training---\nFinal Cross entropy: {evaluate(train_targets, y)[0]} '
          f'\nClassification Accuracy: {evaluate(train_targets, y)[1]}')

    # Validation
    y = logistic(weights, valid_inputs, valid_targets, hyperparameters)[2]
    print(f'---Validation---\nFinal Cross entropy: {evaluate(valid_targets, y)[0]} '
          f'\nClassification Accuracy: {evaluate(valid_targets, y)[1]}')

    # Test
    y = logistic(weights, test_inputs, test_targets, hyperparameters)[2]
    print(f'---Test---\nFinal Cross entropy: {evaluate(test_targets, y)[0]} '
          f'\nClassification Accuracy: {evaluate(test_targets, y)[1]}')

    iterations = np.arange(hyperparameters["num_iterations"])
    plt.title("mnist_train")
    # plt.title("mnist_train_small")
    plt.xlabel("iterations")
    plt.ylabel("cross entropy")
    plt.plot(iterations, entropy_train, label="train")
    plt.plot(iterations, entropy_valid, label="valid")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
