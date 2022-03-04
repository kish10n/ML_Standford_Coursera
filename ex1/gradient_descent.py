# this module will compute values for thetas
# function take X matrix(97*2), y vector(97*1), theta(2*1) alpha and iteration
# 97 is size of dataset
import numpy as np

import cost_function as cf


def gradient_descent(X, y, theta, alpha, iteration):
    m = len(y)  # number of training labels
    J_old = np.zeros((iteration, 1))  # iteration * 1 matrix

    for i in range(iteration):
        theta = theta - alpha * (1 / m) * np.transpose(X).dot((X.dot(theta) - np.transpose([y])))
        # X.dot(theta) is 97*1 matrix and np.transpose([y]) als0 97*1 matrix ([vector]) = matrix (vectorsize*1)

        J_old[i] = cf.cost_function(X, y, theta)

    return theta
