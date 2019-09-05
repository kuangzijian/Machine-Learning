import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints
from drawLine import drawLine


def plotProgressKMeans(X, history_centroids, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of
    #k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.

    plotDataPoints(X, idx, K)
    plt.plot(history_centroids[0:i+1, :, 0], history_centroids[0:i+1, :, 1],
             linestyle='', marker='x', markersize=10, linewidth=3, color='k')
    plt.title('Iteration number {}'.format(i + 1))
    for centroid_idx in range(history_centroids.shape[1]):
        for iter_idx in range(i):
            drawLine(history_centroids[iter_idx, centroid_idx, :], history_centroids[iter_idx + 1, centroid_idx, :])