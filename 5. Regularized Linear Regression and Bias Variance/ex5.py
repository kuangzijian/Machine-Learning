import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from featureNormalize import featureNormalize
from polyFeatures import polyFeatures
from plotFit import plotFit
from validationCurve import validationCurve



# =========== Part 1: Loading and Visualizing Data =============

A = sio.loadmat('ex5data1.mat')
X = A['X']
y = A['y'].ravel()
X_test = A['Xtest']
y_test = A['ytest'].ravel()
X_val = A['Xval']
y_val = A['yval'].ravel()
m = X.shape[0]
m_val = X_val.shape[0]
m_test = X_test.shape[0]

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
#plt.show()


# =========== 1. Regularized Linear Regression ========================
# ========== 1.2 Regularized linear regression cost function ==========

theta = np.array([1, 1])
j, _ = linearRegCostFunction(theta, np.hstack((np.ones((m, 1)), X)), y, 1)

print('Cost at theta = [1 ; 1]:', j)
print('(this value should be about 303.993192)')


# =========== 1.3 Regularized linear regression gradient =============

theta = np.array([1, 1])
_, grad = linearRegCostFunction(theta, np.hstack((np.ones((m, 1)), X)), y, 1)

print('Gradient at theta = [1 ; 1]:', grad.ravel())
print('(this value should be about [-15.303016; 598.250744])')


# ================== 1.4 Fitting linear regression ===================
print('\nPart 4: Train Linear Regression')

# Train linear regression with lambda = 0
l = 0.0
theta = trainLinearReg(np.hstack((np.ones((m, 1)), X)), y, l)

pred = np.hstack((np.ones((m, 1)), X)).dot(theta)

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plt.plot(X, pred, linestyle='--', marker='', color='b')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
#plt.show()


# ===================== 2. Bias-variance ==============================
# ===================== 2.1 Learning curves ===========================

l = 0.0
error_train, error_val = learningCurve(np.hstack((np.ones((m, 1)), X)), y,
                                        np.hstack((np.ones((m_val, 1)), X_val)), y_val, l)

plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
#plt.show()

print('# Training Examples / Train Error / Cross Validation Error')
for i in range(m):
    print('  {0:<19} {1:<13.8f} {2:<.8f}'.format(i + 1, error_train[i], error_val[i]))


# ===================== 3. Polynomial regression ==============================
# ===================== 3.1 Learning Polynomial Regression ====================

p = 8

X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.hstack((np.ones((m, 1)), X_poly))

X_poly_test = polyFeatures(X_test, p)
X_poly_test, dummy_mu, dummy_sigma = featureNormalize(X_poly_test, mu, sigma)
X_poly_test = np.hstack((np.ones((m_test, 1)), X_poly_test))

X_poly_val = polyFeatures(X_val, p)
X_poly_val, dummy_mu, dummy_sigma = featureNormalize(X_poly_val, mu, sigma)
X_poly_val = np.hstack((np.ones((m_val, 1)), X_poly_val))

print('Normalized Training Example 1:')
print(X_poly[0, :])



l = 0.0
theta = trainLinearReg(X_poly, y, l, iteration=500)

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {})'.format(l))
#plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, y_val, l)
plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', marker='v', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(l))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
#plt.show()

# ============= 3.3 Selecting lambda using a cross validation set ============

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, y_val)

plt.figure()
plt.plot(lambda_vec, error_train, color='b', label='Train')
plt.plot(lambda_vec, error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('# lambda / Train Error / Validation Error')
for i in range(len(lambda_vec)):
    print('  {0:<8} {1:<13.8f} {2:<.8f}'.format(lambda_vec[i], error_train[i], error_val[i]))
