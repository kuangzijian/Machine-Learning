import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runKMeans import runKMeans
from kMeansInitCentroids import kMeansInitCentroids

# ================= 1. K-Means Clustering =============================
# ================= 1.1.1 Finding closest centroids ====================
print('Finding closest centroids...')

# Load an example dataset that we will be using
mat_data = sio.loadmat('ex7data2.mat')
X = mat_data['X']

# Select an initial set of centroids
K = 3  # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print(idx[0:3])
print('(the closest centroids should be 0, 2, 1 respectively)')


# ===================== 1.1.2 Computing centroid means =========================
print('Computing centroids means...')

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('(the centroids should be [ 2.428301 3.157924 ], [ 5.813503 2.633656 ], [ 7.119387 3.616684 ])')


# =================== 1.2 K-means on example dataset ======================
print('Running K-Means clustering on example dataset...')

# Load an example dataset
mat_data = sio.loadmat('ex7data2.mat')
X = mat_data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values but in practice you want to generate them automatically,
# such as by settings them to be random examples (as can be seen in kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm
centroids, idx = runKMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.')


# ============= 1.4 Image compression with K-means ===============
print('Running K-Means clustering on pixels from an image.')

# Load an image of a bird
A = imageio.imread('bird_small.png')
A = A.astype(float)/255

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
X = A.reshape([img_size[0] * img_size[1], img_size[2]])

K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly.
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runKMeans(X, initial_centroids, max_iters, 1)


print('Applying K-Means to compress an image.')
idx = findClosestCentroids(X, centroids)

# Recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value.
X_recovered = centroids[idx.astype(int), :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size)

fig = plt.figure()
# Display the original image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(A)
ax1.set_title('Original')
# Display compressed image side by side
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(X_recovered)
ax2.set_title('Compressed, with {} colors.'.format(K))
plt.show()
