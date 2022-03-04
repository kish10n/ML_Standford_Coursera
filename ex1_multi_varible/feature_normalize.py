# feature normalize is a method for data with big difference
import numpy as np


def feature_normalize(X):
    print('first element of training data {:f} and second {:f}\n'.format(X[0, 0], X[0, 1]))
    X_normalized = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        mu[:, i] = np.mean(X[:, i])
        sigma[:, i] = np.std(X[:, i])
        X_normalized[:, i] = (X[:, i] - float(mu[:, i])) / float(sigma[:, i])

    return X_normalized, mu, sigma
