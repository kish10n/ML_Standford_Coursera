# gradient decent for multi variable
import numpy as np

import cost_function_multi as cfm


def gradient_decent_multi(X, y, theta, alpha, iteration):
    # X = 47*3, theta 3*1
    m = len(y)
    J_old = np.zeros((iteration, 1))

    for i in range(iteration):
        theta = theta - alpha * (1 / m) * np.transpose(X).dot((X.dot(theta) - np.transpose([y])))

        J_old[i] = cfm.cost_function_multi(X, y, theta)
    return theta, J_old
