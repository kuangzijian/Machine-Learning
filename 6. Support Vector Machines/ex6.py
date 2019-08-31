# Exercise 6-1: Support Vector Machines
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

from plotData import plotData
from gaussianKernel import gaussianKernel
from visualizeBoundaryLinear import visualizeBoundaryLinear
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params


# =============== 1. Support Vector Machines ================
print('Loading and Visualizing Dataset 1...')

# Load from ex6data1
mat_data = sio.loadmat('ex6data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Plot training data
plt.figure()
plotData(X, y)
plt.xlim([0, 4.5])
plt.ylim([1.5, 5])
plt.title("Dataset 1")
plt.show()


# ==================== 1.1 Example dataset 1 ====================
# Load from ex6data1
mat_data = sio.loadmat('ex6data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

print('Training Linear SVM...')
# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
C = 1
clf = svm.LinearSVC(C=C)
clf.fit(X, y)
print('score:', clf.score(X, y))

plt.figure()
visualizeBoundaryLinear(X, y, clf)
plt.xlim([0, 4.5])
plt.ylim([1.5, 5])
plt.title("Dataset 1")
plt.show()


# =============== 1.2.1 Gaussian kernel ===============
print('Evaluating the Gaussian Kernel...')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {}: \n\t{}\n' \
      '(for sigma = 2, this value should be about 0.324652)'.format(sigma, sim))


# =============== 1.2.2 Example dataset 2 ================
print('Loading and Visualizing Dataset 2...')

# Load from ex6data2
mat_data = sio.loadmat('ex6data2.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Plot training data
plt.figure()
plotData(X, y)
plt.xlim([0, 1])
plt.ylim([0.4, 1])
plt.title("Dataset 2")
plt.show()


# ========== 1.2.3 Example dataset 3 ==========
print('Training SVM with RBF Kernel...')
# Load from ex6data1
mat_data = sio.loadmat('ex6data2.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# SVM Parameters
C = 100
gamma = 10

clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
clf.fit(X, y)
print('score:', clf.score(X, y))

plt.figure()
visualizeBoundary(X, y, clf)
plt.xlim([0, 1])
plt.ylim([0.4, 1])
plt.title("Dataset 2")
plt.show()

print('Loading and Visualizing Data 3...')

# Load from ex6data3
mat_data = sio.loadmat('ex6data3.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Plot training data
plt.figure()
plotData(X, y)
plt.xlim([-0.6, 0.3])
plt.ylim([-0.7, 0.6])
plt.title("Dataset 3")
plt.show()

# Load from ex6data3
mat_data = sio.loadmat('ex6data3.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

# Try different SVM Parameters here
C, gamma = dataset3Params(X, y, X_val, y_val)

# Train the SVM
clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
clf.fit(X, y)
print('score:', clf.score(X_val, y_val))

plt.figure()
visualizeBoundary(X, y, clf)
plt.xlim([-0.6, 0.3])
plt.ylim([-0.7, 0.6])
plt.title("Dataset 3")
plt.show()
