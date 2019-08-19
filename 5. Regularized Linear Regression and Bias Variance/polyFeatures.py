import numpy as np


def polyFeatures(X, p):
    #POLYFEATURES Maps X (1D vector) into the p-th power
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #

    X_poly = np.zeros((len(X), p))

    for i in range(p):
        X_poly[:, i] = np.power(X, i + 1).ravel()

    return X_poly