import numpy as np

# NORMALEQN Computes the closed-form solution to linear regression
#    NORMALEQN(X,y) computes the closed-form solution to linear
#    regression using the normal equations.

#  ====================== YOUR CODE HERE ======================
#  Instructions: Complete the code to compute the closed form solution
#                to linear regression and put the result in theta.
# 

#  ---------------------- Sample Solution ----------------------

def normalEqn(X, y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
