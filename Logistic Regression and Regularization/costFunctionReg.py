import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, l):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # calculates hypothesis
    z = X.dot(theta)
    h = sigmoid(z)

    newTheta = theta.copy()
    newTheta[0] = 0

    J = (1 / m) * (np.log(h).T.dot(-y) - np.log(1-h).T.dot(1-y)) + (l/(2 * m))*np.sum(np.power(newTheta, 2))

    grad = (1.0 / m) * (X.T.dot(h - y)) + 1.0 * (l/m) * newTheta

    return J, grad




