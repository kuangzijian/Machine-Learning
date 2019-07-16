import numpy as np
from computeCost import computeCost

import matplotlib.pyplot as plt

#%   GRADIENTDESCENT Performs gradient descent to learn theta
#%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
#%   taking num_iters gradient steps with learning rate alpha

#% Initialize some useful values

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y) #% number of training examples
    J_history = np.zeros(num_iters)

    for iter in range(1, num_iters):
        theta -= alpha / m * ((X.dot(theta) - y).T.dot(X))
        #% Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)



    return theta

