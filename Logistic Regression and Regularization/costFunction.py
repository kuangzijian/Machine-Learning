import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    # calculates hypothesis
    z = X.dot(theta)
    h = sigmoid(z)

    # J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h))
    J = (1 / m) * (np.log(h).T.dot(-y) - np.log(1-h).T.dot(1-y))
    # grad = (1 / m) * sum((h - y) .* X)'
    grad = (1 / m) * (X.T.dot(h - y))

    return J, grad



