import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #    trained weights of a neural network (Theta1, Theta2)

    #  Useful values
    m, n = X.shape

    #  You need to return the following variables correctly
    #  ====================== YOUR CODE HERE ======================
    #  Instructions: Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                vector containing labels between 1 to num_labels.
    #
    #  Hint: The max function might come in useful. In particular, the max
    #        function can also return the index of the max element, for more
    #        information see 'help max'. If your examples are in rows, then, you
    #        can use max(A, [], 2) to obtain the max for each row.
    #

    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = np.hstack((np.ones((m, 1)), sigmoid(a1.dot(Theta1.T))))
    a3 = sigmoid(a2.dot(Theta2.T))
    p = np.argmax(a3, axis=1)
    p += 1

    return p