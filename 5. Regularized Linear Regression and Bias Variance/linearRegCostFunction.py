import numpy as np

def linearRegCostFunction(theta, X, y, l):
    #LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    #regression with multiple variables
    #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    #   cost of using theta as the parameter for linear regression to fit the
    #   data points in X and y. Returns the cost in J and the gradient in grad

    # Initialize some useful values
    m = X.shape[0] # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    h = X.dot(theta)
    squaredErrors = np.square(h - y)
    thetaExcludingZero = theta.copy()
    thetaExcludingZero[0:1] = 0

    J = 1.0 / (2 * m) * np.sum(squaredErrors) + 1.0 * l / (2 * m) * np.sum(np.square(theta[1:]))
    grad = 1.0 / m * X.T.dot(h - y) + 1.0 * l / m * thetaExcludingZero

    return J, grad
