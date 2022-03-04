# This is a first exercise of ML Course, Standford
# Lineare one Variable
# hypothesis h(x) = theta0 + theta1*x
# cost function = 1/(2*m)*sum(h(x)-y)^2

import matplotlib.pyplot as plt
# np.loadtxt from numpy library to load data file.
import numpy as np
from matplotlib import cm

import cost_function as cf
import gradient_descent as gd
import plot_data as pd

data = np.loadtxt('ex1data.txt', delimiter=",")
x_data = data[:, 0]  # first raw, starts with 0
y = data[:, 1]  # second raw

m = len(y)  # get length of vector y

pd.plot_data(x_data, y)

X = np.column_stack((np.ones((m, 1)), x_data))  # np.ones add 1, m is size of column and 1 is how many times 1

theta = np.zeros((2, 1))  # create a 2*1 matrix with zeros. theta0 and theta1

# theta optimization through gradient descent
# alpha and iteration
alpha = 0.01
iteration = 1500

print(cf.cost_function(X, y, theta))

theta = gd.gradient_descent(X, y, theta, alpha, iteration)  # is 2*1 matrix

plt.plot(X, theta[0, 0] + theta[1, 0] * X)
print('Theta found by gradient descent: ')
print("{:f}, {:f}".format(theta[0, 0], theta[1, 0]))

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, we predict a profit of {:f}".format(float(predict1 * 10000)))
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of {:f}'.format(float(predict2 * 10000)))

input('Program paused. Press enter to continue.\n')
# Visualizing J(theta0, theta1....)
theta0_values = np.linspace(-10, 10, 100)  # linspace will create sequence, create 100 sequence between -10 to 10
theta1_values = np.linspace(-1, 4, 100)
J_values = np.zeros(((len(theta0_values)), len(theta1_values)))
for i in range(len(theta0_values)):
    for j in range(len(theta1_values)):
        t = [[theta0_values[i]], [theta1_values[j]]]
        J_values[i, j] = cf.cost_function(X, y, t)

J_values = np.transpose(J_values)

# Contour plot

figure = plt.figure()
axis = figure.add_subplot(111)
cset = plt.contour(theta0_values, theta1_values, J_values, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
figure.colorbar(cset)
plt.xlabel('theta0_values')
plt.ylabel('theta1_values')
plt.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
plt.show()
