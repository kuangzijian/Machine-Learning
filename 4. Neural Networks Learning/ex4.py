import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt

from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from predict import predict

# ==================== 1. Neural Networks  ====================
# Load saved matrices from file

mat_data = sio.loadmat('ex4data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
m, n = X.shape

input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
plt.figure()
displayData(X[rand_indices[0:100], :], padding=1)
plt.show()

# =================== 1.2 Model representation ===================
# Load the weights into variables Theta1 and Theta2
mat_param = sio.loadmat('ex4weights.mat')
theta_1 = mat_param['Theta1']
theta_2 = mat_param['Theta2']

params_trained = np.hstack((theta_1.flatten(), theta_2.flatten()))

# =================== 1.3 Feedforward and cost function =============
l = 0.0
j, _ = nnCostFunction(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print('Cost at parameters (this value should be about 0.287629):', j)

# =================== 1.4 Regularized cost function ==================

l = 1.0
j, _ = nnCostFunction(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print('Cost at parameters (this value should be about 0.383770):', j)

# ==================== 2. Backpropagation ============================

# ==================== 2.1 Sigmoid gradient ==========================
print('Evaluating sigmoid gradient...')
g = sigmoidGradient(0)
print(g)

# ==================== 2.2 Random initialization =====================

initial_theta_1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta_2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.hstack((initial_theta_1.ravel(), initial_theta_2.ravel()))

# ==================== 2.3 Backpropagation ===========================
l = 3.0
debug_j, _ = nnCostFunction(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print('Cost at (fixed) debugging parameters (w/ lambda = {}): {}'.format(l, debug_j))
print('(for lambda = 3, this value should be about 0.576051)')

# ==================== 2.6 Learning parameters using scipy.optimize.minimize ==========
l = 1.0
result = opt.minimize(fun=nnCostFunction, x0=initial_nn_params,
                      args=(input_layer_size, hidden_layer_size, num_labels, X, y, l),
                      method='tnc', jac=True, options={'maxiter': 150})
params_trained = result.x
Theta_1_trained = np.reshape(params_trained[0:(hidden_layer_size * (input_layer_size + 1)), ],
                             (hidden_layer_size, input_layer_size + 1))
Theta_2_trained = np.reshape(params_trained[(hidden_layer_size * (input_layer_size + 1)):, ],
                             (num_labels, hidden_layer_size + 1))

# ==================== 3. Visualizing the hidden layer ========================
plt.figure()
displayData(Theta_1_trained[:, 1:], padding=1)
plt.show()

# ================= 3.1 Optional (ungraded) exercise: predict =================
pred = predict(Theta_1_trained, Theta_2_trained, X)
print('Training Set Accuracy:', np.mean(pred == y) * 100)

rp = np.random.permutation(m)
for i in range(m):
    print('Displaying Example Image')
    displayData(X[rp[i],].reshape(1, n))

    pred = predict(Theta_1_trained, Theta_2_trained, X[rp[i],].reshape(1, n))
    print('Neural Network Prediction: {} (digit {})'.format(pred, pred % 10))
    plt.show()
    s = input('Paused - press enter to continue, q to exit: ')
    if s == 'q':
        break