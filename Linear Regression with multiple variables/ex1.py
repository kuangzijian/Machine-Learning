import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

A = np.loadtxt(open('ex1data1.txt', 'r'), delimiter=",")
A = np.array(A)

x = np.transpose(A[:,0]).reshape(-1,1)
y = np.transpose(A[:,1]).reshape(-1,1)

plotData(x,y)

m = len(x)
x0 = np.ones((m,1))
X = np.hstack((x0,x))
theta = np.zeros(2).reshape(-1,1)
iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print(J)

theta = gradientDescent(X, y, theta, alpha, iterations)
print('Theta computed from gradient descent: ', theta[0], theta[1])

plt.figure()
plotData(list(x),list(y))
plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
plt.legend(loc='upper left', numpoints=1)
plt.show()


predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, we predict a profit of", predict1 * 10000)

predict2 = np.array([1, 7]).dot(theta)
print("For population = 70,000, we predict a profit of", predict2 * 10000)