import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize

# ============= 3. Linear regression with multiple variables =============
A = np.loadtxt(open('ex1data2.txt', 'r'), delimiter=",")
A = np.array(A)

X = A[:, 0:2]
y = A[:, 2]
m = len(y)

# ============= 3.1 Feature Normalization =============
X, mu, sigma = featureNormalize(X)

#Add intercept term to X
x0 = np.ones((m, 1))
X = np.hstack((x0, X))

# ============= 3.2 Gradient Descent =============

# Run gradient descent
# Choose some alpha value
alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros(3)
theta, _ = gradientDescent(X, y, theta, alpha, num_iters)

# Display gradient descent's result
print('Theta computed from gradient descent:', theta[0], theta[1], theta[2])
