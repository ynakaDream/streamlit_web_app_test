import numpy as np


def  least_squares_method(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    a = np.sum( (x - x_mean) * (y - y_mean) ) / np.sum((x - x_mean) ** 2)
    b = y_mean - a * x_mean

    return a, b


def gaussian(x, mu=0.0, sigma=1.0):
    x = np.asarray(x)
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2/(2*sigma**2))
