import numpy as np
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from normalEqn import normalEqn

# ============= 3. Linear regression with multiple variables =============
A = np.loadtxt(open('ex1data2.txt', 'r'), delimiter=",")
A = np.array(A)

X = A[:, 0:2]
y = A[:, 2]
m = len(y)

# ============= 3.1 Feature Normalization =============
X, mu, sigma = featureNormalize(X)

# Add intercept term to X
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

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

test = np.array([1650, 3])
test = (test - mu) / sigma
test = np.hstack((np.ones(1), test))

price = test.dot(theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent)', price)


# ================ 3.3 Normal Equations ================

A = np.loadtxt(open('ex1data2.txt', 'r'), delimiter=',')
A = np.array(A)

X = A[:, 0:2]
y = A[:, 2]
m = len(y)

# Add intercept term for X

X = np.hstack((np.ones((m,1)), X))

# Calculate the parameters from the normal equation

theta = normalEqn(X, y)
print('Theta computed from normal equations:', theta[0], theta[1], theta[2])

price = np.array([1, 1650, 3]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)

