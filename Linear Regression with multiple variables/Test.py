import numpy as np
from plotData import plotData

A = []
with open('ex1data1.txt', 'r') as f:
    for line in f:
        A.append(list(map(float,line.split(','))))
A = np.array(A)

a = list(A[:,0])
b = list(A[:,1])

plotData(a,b)