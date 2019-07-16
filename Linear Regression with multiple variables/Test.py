import numpy as np
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
import matplotlib.pyplot as plt

A = np.loadtxt(open('ex1data1.txt', 'r'), delimiter=",")
A = np.array(A)

x = A[:,0]
y = A[:,1]

plotData(x,y)

m = len(x)

x0 = np.ones((m,1))
X = np.hstack((x0,x.reshape(-1,1)))
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print(J)

theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

plt.figure()
plotData(list(x),list(y))
plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
plt.legend(loc='upper left', numpoints=1)
plt.show()