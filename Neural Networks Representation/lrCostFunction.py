import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, l):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y) # number of training examples
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation

    #           sigmoid(X * theta)

    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.

    # Hint: When computing the gradient of the regularized cost function,
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta;
    #           temp(1) = 0;   # because we don't add anything for j = 0
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)

    # calculates hypothesis
    z = X.dot(theta)
    h = sigmoid(z)

    newTheta = theta.copy()
    newTheta[0] = 0

    J = (1 / m) * (np.log(h).T.dot(-y) - np.log(1-h).T.dot(1-y)) + (l/(2 * m))*np.sum(np.power(newTheta, 2))

    grad = (1.0 / m) * (X.T.dot(h - y)) + 1.0 * (l/m) * newTheta

    return J, grad




