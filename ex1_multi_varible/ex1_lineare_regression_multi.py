# This is a first exercise of ML Course, Standford
# Lineare Regression with multiple Variable
# hypothesis h(x) = theta0 + theta1*x1 + theta2*x2
# cost function = 1/(2*m)*sum(h(x)-y)^2

# load a data
import numpy as np
from matplotlib import pyplot as plt

import feature_normalize as fn
import gradient_descent_multi as gdm
import normal_eqn as ne

data = np.loadtxt('ex1data2.txt', delimiter=',')

x_data = data[:, :2]  # matrix of size of training data * 2
y = data[:, 2]  # matrix of training data * 1
m = len(y)  # size of training data is 47

X_normalized, mu, sigma = fn.feature_normalize(x_data)
X = np.column_stack((np.ones((m, 1)), X_normalized))  # 47*3 first column is 1, 2 and 3 are training data.

alpha = 0.01
iteration = 400

theta = np.zeros((3, 1))  # 3*1 matrix

theta, J_old = gdm.gradient_decent_multi(X, y, theta, alpha, iteration)
print('from gradient decent, {}'.format(theta))

plt.plot(range(J_old.size), J_old, "-b", linewidth=2)
plt.xlabel('Number of iteration')
plt.ylabel('Cost J')
plt.show()

# normal equations

theta = ne.normal_equations(X, y)
print('from normal equation, {:f}, {:f}, {:f}'.format(theta[0], theta[1], theta[2]))


