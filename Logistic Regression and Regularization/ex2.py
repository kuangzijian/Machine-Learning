import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import plotData
from costFunction import costFunction
from sigmoid import sigmoid


# Load Data
# The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt(open("ex2data1.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]


# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plt.figure()
plotData(X, y)
pos = np.argwhere(y == 1)
neg = np.argwhere(y == 0)
plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.xlim([30, 100])
plt.ylim([30, 100])
plt.legend(['Admitted', 'Not admitted'], loc='upper right', numpoints=1)
plt.show()


# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.hstack((np.ones((m, 1)), X))

# Initialize fitting parameters
theta = np.zeros(n + 1)  # Initialize fitting parameters

cost, grad = costFunction(theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)
