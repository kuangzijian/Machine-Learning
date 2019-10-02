import numpy as np
import matplotlib.pyplot as plt

from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    #VISUALIZEFIT Visualize the dataset and its estimated distribution.
    #   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
    #   probability density function of the Gaussian distribution. Each example
    #   has a location (x1, x2) that depends on its feature values.
    #
    l = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(l, l)

    X_tmp = np.vstack((X1.ravel(), X2.ravel())).T
    Z = multivariateGaussian(X_tmp, mu, sigma2)
    Z.resize(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))