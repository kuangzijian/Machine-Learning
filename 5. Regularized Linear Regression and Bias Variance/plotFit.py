import numpy as np
import matplotlib.pyplot as plt

from polyFeatures import polyFeatures
from featureNormalize import featureNormalize


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Parameters
    ----------
    min_x : float
        Minimum value of features.
    max_x : float
        Maximum value of features.
    mu : ndarray, shape (n_features - 1,)
        Mean value of features, without the intercept term.
    sigma : ndarray, shape (n_features - 1,)
        Standard deviation of features, without the intercept term.
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    p : int
        Power of polynomial fit.
    """
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    X_poly = polyFeatures(x, p)
    X_poly, dummy_mu, dummy_sigma = featureNormalize(X_poly, mu, sigma)
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
    plt.plot(x, X_poly.dot(theta), linestyle='--', marker='', color='b')