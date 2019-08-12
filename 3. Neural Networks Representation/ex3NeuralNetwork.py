import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from displayData import displayData
from predict import predict

# ===================== 2. Neural Networks ==============================

# Load saved matrices from file
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10

data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].ravel()
m, n = X.shape

# =================== 2.1 Model representation ===========================
# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]
plt.figure()
displayData(sel, padding=1)
plt.show()

mat_param = sio.loadmat('ex3weights.mat')
Theta_1 = mat_param['Theta1']
Theta_2 = mat_param['Theta2']

# =================== 2.2 Feedforward propagation and prediction ==========
pred = predict(Theta_1, Theta_2, X)
print('Training Set Accuracy:', np.mean(pred == y) * 100)

rp = np.random.permutation(m)
for i in range(m):
    print('Displaying Example Image')
    displayData(X[rp[i],].reshape(1, n))

    pred = predict(Theta_1, Theta_2, X[rp[i],].reshape(1, n))
    print('Neural Network Prediction: {} (digit {})'.format(pred, pred % 10))
    plt.show()
    s = input('Paused - press enter to continue, q to exit: ')
    if s == 'q':
        break