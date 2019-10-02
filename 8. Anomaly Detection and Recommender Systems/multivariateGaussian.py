import numpy as np


def multivariateGaussian(X, mu, Sigma2):
    #MULTIVARIATEGAUSSIAN Computes the probability density function of the
    #multivariate gaussian distribution.
    #    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
    #    density function of the examples X under the multivariate gaussian
    #    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    #    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    #

    k = len(mu)

    if len(Sigma2.shape) == 1:
        Sigma2 = np.diag(Sigma2)

    X_mu = X - mu
    p = (2 * np.pi) ** (-k / 2.0) * np.linalg.det(Sigma2) ** (-0.5) \
        * np.exp(-0.5 * np.sum(X_mu.dot(np.linalg.pinv(Sigma2)) * X_mu, axis=1))

    return p