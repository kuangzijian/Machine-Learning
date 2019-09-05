import numpy as np
import matplotlib.pyplot as plt

from plotProgressKMeans import plotProgressKMeans
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids

def runKMeans(X, initial_centroids, max_iters, plot_progress):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a true/false flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to false by default. runkMeans returns
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    history_centroids = np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
    idx = np.zeros(X.shape[0])

    for i in range(max_iters):
        print('K-Means iteration {}/{}'.format(i + 1, max_iters))
        history_centroids[i, :] = centroids

        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            plt.figure()
            plotProgressKMeans(X, history_centroids, idx, K, i)
            plt.show()

        centroids = computeCentroids(X, idx, K)

    return centroids, idx
