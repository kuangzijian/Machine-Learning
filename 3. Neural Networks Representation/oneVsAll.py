import numpy as np
import scipy.optimize as opt
from lrCostFunction import lrCostFunction

# ONEVSALL trains multiple logistic regression classifiers and returns all
# the classifiers in a matrix all_theta, where the i-th row of all_theta 
# corresponds to the classifier for label i
#    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#    logistic regression classifiers and returns each of these classifiers
#    in a matrix all_theta, where the i-th row of all_theta corresponds 
#    to the classifier for label i

def oneVsAll(X, y, num_labels, l):
#  Some useful variables
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))
    initial_theta = np.zeros(n + 1)

    #  ====================== YOUR CODE HERE ======================
    #  Instructions: You should complete the following code to train num_labels
    #                logistic regression classifiers with regularization
    #                parameter lambda.
    #
    #  Hint: theta(:) will return a column vector.
    #
    #  Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    #        whether the ground truth is true/false for this class.
    #
    #  Note: For this assignment, we recommend using fmincg to optimize the cost
    #        function. It is okay to use a for-loop (for c = 1:num_labels) to
    #        loop over the different classes.
    #
    #        fmincg works similarly to fminunc, but is more efficient when we
    #        are dealing with large number of parameters.

    for i in range(0, num_labels):
        label = 10 if i == 0 else i
        result = opt.minimize(fun=lrCostFunction, x0=initial_theta, args=(X, (y == label).astype(int), l), method='TNC', jac=True)
        all_theta[i, :] = result.x


    return all_theta
