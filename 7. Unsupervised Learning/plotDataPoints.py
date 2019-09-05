import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    #   with the same index assignments in idx have the same color
    color = cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(X[:, 0], X[:, 1], c=color[idx.astype(int), :])