import numpy as np
import numpy.matlib

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1)), ],
                         (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):, ],
                         (num_labels, hidden_layer_size + 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))

    a1 = X
    a2 = np.hstack((np.ones((m, 1)), sigmoid(a1.dot(Theta1.T))))
    h = sigmoid(a2.dot(Theta2.T))

    # Constructing a vector of result ex: for 5 of 10 the 1 should be at
    # fifth position [0 0 0 0 1 0 0 0 0 0] where rows are training set samples
    yVec = np.equal(np.matlib.repmat(list(range(1, 11)), m, 1), np.matlib.repmat(y, num_labels, 1).T).astype(np.int)

    # Cost Function
    cost = -yVec * np.log(h) - (1 - yVec) * np.log(1 - h)
    J = (1 / m) * sum(sum(cost))

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    theta1ExcludingBias  = Theta1[:, 1:]
    theta2ExcludingBias  = Theta2[:, 1:]
    reg = 1.0 * l / (2 * m) * (sum(sum(np.square(theta1ExcludingBias))) + sum(sum(np.square(theta2ExcludingBias))))

    J = J + reg

    d3 = h - yVec
    D2 = d3.T.dot(a2)

    Z2 = np.hstack((np.ones((m, 1)), a1.dot(Theta1.T)))
    d2 = d3.dot(Theta2) * sigmoidGradient(Z2)
    d2 = d2[:, 1:]
    D1 = d2.T.dot(X)

    Theta_1_grad = 1.0 * D1 / m


    Theta_2_grad = 1.0 * D2 / m

    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    Theta_1_grad[:, 1:] = Theta_1_grad[:, 1:] + 1.0 * l / m * Theta1[:, 1:]
    Theta_2_grad[:, 1:] = Theta_2_grad[:, 1:] + 1.0 * l / m * Theta2[:, 1:]

    # Unroll gradients
    grad = np.hstack((Theta_1_grad.ravel(), Theta_2_grad.ravel()))

    return J, grad