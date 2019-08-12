import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from costFunction import costFunction
from sigmoid import sigmoid
from predict import predict

# ==================== 1. Logistic Regression  ====================

# Load Data
# The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt(open("ex2data1.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]


# ==================== 1.1 Visualizing the data ====================
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

# ============ 1.2.2 Cost function and gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.hstack((np.ones((m, 1)), X))

# Initialize fitting parameters
theta = np.zeros(n + 1)  # Initialize fitting parameters

cost, grad = costFunction(theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad)

# ============= 1.2.3 Learning parameters using fminunc  =============
theta, nfeval, rc = opt.fmin_tnc(func=costFunction, x0=theta, args=(X, y), messages=0)
if rc == 0:
    print('Local minimum reached after', nfeval, 'function evaluations.')

# Print theta to screen
cost, _ = costFunction(theta, X, y)
print('Cost at theta found by fminunc:', cost)
print('theta:', theta)

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.show()

# ============== 1.2.4 Evaluating logistic regression ==============
# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print('For a student with scores 45 and 85, we predict an admission probability of', prob)

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy:', np.mean(p == y) * 100)


