import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# ============= 2.1 Plotting the data =============
A = np.loadtxt(open('ex1data1.txt', 'r'), delimiter=",")
A = np.array(A)

x = A[:,0]
y = A[:,1]
m = len(x)

# ============= 2.2 Gradient Descent =============

#Add intercept term to X
x0 = np.ones((m,1))
X = np.hstack((x0,x.reshape(m, 1)))
theta = np.zeros(2)
num_iters = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print(J)

theta, _ = gradientDescent(X, y, theta, alpha, num_iters)
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

# ============= 2.4 Visualizing  =============

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out j_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)

# We need to transpose J_vals before calling plot_surface, or else the axes will be flipped.
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=2, cstride=2, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')

# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, base=20))
plt.plot(theta[0], theta[1], linestyle='', marker='x', color='r')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

plt.show()
