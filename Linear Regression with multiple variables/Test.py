import numpy as np
from plotData import plotData
from computeCost import computeCost

A = []
with open('ex1data1.txt', 'r') as f:
    for line in f:
        A.append(list(map(float,line.split(','))))
A = np.array(A)

x = A[:,0]
y = A[:,1]

#plotData(list(x),list(y))

m = len(x)
x = x.reshape((-1,1))
y = y.reshape((-1,1))
#Add a column of ones to x
x0 = np.ones((m,1))
X = np.hstack((x0,x))
theta = np.array([[0],[0]])
iterations = 1500
alpha = 0.01


J = computeCost(X, y, theta)

print(J)
