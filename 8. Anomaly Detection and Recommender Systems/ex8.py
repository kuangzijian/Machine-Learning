import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt

from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold
from cofiCostFunc import cofiCostFunc
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings

# ================== 1. Anomaly Detection  ===================
mat_data = sio.loadmat('ex8data1.mat')
X = mat_data['X']
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

plt.figure()
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== 1.2 Estimating parameters for a Gaussian ===================
# Estimate mu and sigma2
mu, sigma2 = estimateGaussian(X)

# Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2)

# Visualize the fit
plt.figure()
visualizeFit(X, mu, sigma2)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== 1.3 Selecting the threshold ===================
p_val = multivariateGaussian(X_val, mu, sigma2)

epsilon, F1 = selectThreshold(y_val, p_val)

print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
print('(you should see a value epsilon of about 8.99e-05)')

outliers = np.nonzero(p < epsilon)

plt.figure()
visualizeFit(X, mu, sigma2)
plt.scatter(X[outliers, 0], X[outliers, 1], facecolors='none', edgecolors='r', s=100)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== 1.4 High dimensional dataset ===================
# Loads the second dataset. You should now have the variables X, Xval, yval in your environment
mat_data = sio.loadmat('ex8data2.mat')
X = mat_data['X']
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

# Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

# Training set
p = multivariateGaussian(X, mu, sigma2)

# Cross-validation set
p_val = multivariateGaussian(X_val, mu, sigma2)

# Find the best threshold
epsilon, F1 = selectThreshold(y_val, p_val)

print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
print('# Outliers found:', np.sum(p < epsilon))
print('(you should see a value epsilon of about 1.38e-18)')


# =============== 2. Recommender Systems ================
mat_data = sio.loadmat('ex8_movies.mat')

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
Y = mat_data['Y']

# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = mat_data['R']

print('Average rating for movie 1 (Toy Story): {} / 5'.format(np.mean(Y[0, np.nonzero(R[0, ])])))

# We can "visualize" the ratings matrix by plotting it with imshow
plt.figure()
plt.imshow(Y, aspect='auto')
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()


# ============ 2.2 Collaborative filtering learning algorithm ===========
# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
mat_data = sio.loadmat('ex8_movieParams.mat')
X = mat_data['X']
Theta = mat_data['Theta']
num_users = mat_data['num_users'].ravel()[0]
num_movies = mat_data['num_movies'].ravel()[0]
num_features = mat_data['num_features'].ravel()[0]

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

J, grad = cofiCostFunc(np.hstack((X.flatten(), Theta.flatten())), Y, R, num_users, num_movies, num_features, 0)
print('Cost at loaded parameters:', J)
print('(this value should be about 22.22)')


# ========= Part 4: Collaborative Filtering Cost Regularization ========
# Evaluate cost function

J, grad = cofiCostFunc(np.hstack((X.flatten(), Theta.flatten())), Y, R, num_users, num_movies, num_features, 1.5)
print('Cost at loaded parameters (lambda = 1.5):', J)
print('(this value should be about 31.34)')


# ============== 2.3 Learning movie recommendations ===============
movie_list = loadMovieList()

# Initialize my ratings
my_ratings = np.zeros(len(movie_list), dtype=np.int)
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('New user ratings:')
for i in np.argwhere(my_ratings > 0).ravel():
    print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

print('Training collaborative filtering...')
mat_data = sio.loadmat('ex8_movies.mat')

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
Y = mat_data['Y']
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = mat_data['R']

# Add our own ratings to the data matrix
Y = np.hstack((my_ratings.reshape(len(movie_list), 1), Y))
R = np.hstack((my_ratings.reshape(len(movie_list), 1) != 0, R))

# Normalize Ratings
Y_norm, Y_mean = normalizeRatings(Y, R)

# Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.hstack((X.flatten(), Theta.flatten()))

# Set Regularization
l = 10
result = opt.minimize(fun=cofiCostFunc, x0=initial_parameters,
                      args=(Y_norm, R, num_users, num_movies, num_features, l),
                      method='CG', jac=True, options={'maxiter': 150})

X = result.x[0:num_movies * num_features].reshape((num_movies, num_features))
Theta = result.x[num_movies * num_features:].reshape((num_users, num_features))

print('Recommender system learning completed.')


# ================== 2.3.1 Recommendations ====================
p = X.dot(Theta.T)
my_predictions = p[:, 0] + Y_mean
idx = np.argsort(my_predictions)[::-1]
print('Top recommendations for you:')
for i in range(10):
    print('Predicting rating {0:.1f} for movie {1:s}'.format(my_predictions[idx[i]], movie_list[idx[i]]))

print('Original ratings provided:')
for i in np.argwhere(my_ratings > 0).ravel():
    print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))