import numpy as np


def pca(X):
#PCA Run principal component analysis on the dataset X
#   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
#   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
#
    m, n = X.shape
    sigma = X.T.dot(X) / m
    U, S, V = np.linalg.svd(sigma)
    return U, S, V