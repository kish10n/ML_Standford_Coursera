# to compute cost function
# it takes X matrix(97*2), y vector(97*1), theta(2*1)
# 97 is size of dataset
import numpy as np


def cost_function(X, y, theta):
    m = len(y)
    compute_cost = np.power(X.dot(theta) - np.transpose([y]), 2)
    J = 1 / (2 * m) * compute_cost.sum(axis=0)

    return J
