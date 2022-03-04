# alternative to find theta, normal equations
import numpy as np


def normal_equations(X, y):
    # for normal equation is a formal, invers(transpose(matrix)*matrix)*transpose(matrix)*y

    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

    return theta
