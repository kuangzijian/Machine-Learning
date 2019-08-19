import numpy as np
import scipy.optimize as opt

from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, l, iteration=200):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    m, n = X.shape
    initial_theta = np.zeros((n, 1))

    # Minimize using fmincg
    result = opt.minimize(fun=linearRegCostFunction, x0=initial_theta, args=(X, y, l), method='TNC', jac=True,
                          options={'maxiter': iteration})

    return result.x

