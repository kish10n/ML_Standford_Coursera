# plot_data plots the data points x and y into new figure
from matplotlib import pyplot as plt


def plot_data(X, y):
    plt.plot(X, y, '+', markersize=10, label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show(block=False)
