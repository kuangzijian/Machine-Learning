import numpy as np


def kMeansInitCentroids(X, K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    #used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #

    rand_idx = np.random.permutation(X.shape[0])
    centroids = X[rand_idx[0:K], :]
    return centroids