import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

# ==================== 2. Regularized logistic regression  ====================

# Load Data
# The first two columns contains the X values and the third column contains the label (y).
data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]

# ==================== 2.1 Visualizing the data ====================
plt.figure()
#plotData(X, y)
pos = np.argwhere(y == 1)
neg = np.argwhere(y == 0)
plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right', numpoints=1)

# =========== 2.2 Feature mapping ============

# Add Polynomial Features
# Note that map_feature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X[:, 0], X[:, 1])
m, n = X.shape

# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to 1
l = 1.0

# ============ 2.3 Cost function and gradient ==========
cost, _ = costFunctionReg(initial_theta, X, y, l)

print('Cost at initial theta (zeros):', cost)


# ============= 2.3.1 Learning parameters using fminunc =============
# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to t1 (you should vary this)
l = 1.0

# Optimize
theta, nfeval, rc = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(X, y, l), messages=0)

# ================= 2.4 Plotting the decision boundary ==============
# Plot Boundary
#plotData(X[:, 1:], y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right', numpoints=1)

plotDecisionBoundary(theta, X, y)
plt.show()

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy:', np.mean(p == y) * 100)
