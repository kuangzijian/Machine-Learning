import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from displayData import displayData
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

# ==================== 1. Multi-class Classification  ====================
# Load saved matrices from file
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10

data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].ravel()
m, n = X.shape

# =================== 1.2 Visualizing the data ===========================
# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]
plt.figure()
displayData(sel, padding=1)
plt.show()

# ===================== 1.3 Vectorizing logistic regression ==============
theta_t = np.array([-2, -1, 1, 2])
X_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0))
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

cost, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost:', cost)
print('Expected cost: 2.534819')
print('Gradients: \n', grad)
print('Expected gradients: \n  [ 0.146561 -0.548558 0.724722 1.398003 ]')

# =================== 1.4 One-vs-all classication ========================

l = 0.1
all_theta = oneVsAll(X, y, num_labels, l)

# ==================== 1.4.1 One-vs-all prediction=========================

pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy:', np.mean(pred == y) * 100)
