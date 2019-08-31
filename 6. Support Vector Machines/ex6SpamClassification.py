# Exercise 6-2: Spam Classification with SVMs
import numpy as np
import scipy.io as sio
from sklearn import svm

from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocablist import getVocablist


# ==================== 2. Spam Classification ====================
print('Preprocessing sample email (emailSample1.txt)...')

with open('emailSample1.txt') as f:
    file_contents = f.read().replace('\n', '')

word_indices = processEmail(file_contents)

# print(Stats
print('Word Indices:', word_indices)


# ==================== 2.2 Extracting features from emails ====================
print('Extracting features from sample email (emailSample1.txt)...')
features = emailFeatures(word_indices)

print('Length of feature vector:', len(features))
print('Number of non-zero entries:', np.sum(features > 0))


# =========== 2.3 Training SVM for spam classification ========
# Load the Spam Email dataset
mat_data = sio.loadmat('spamTrain.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

print('Training Linear SVM (Spam Classification)...')
C = 0.1
clf = svm.LinearSVC(C=C)
clf.fit(X, y)
p = clf.predict(X)

print('Training Accuracy:', np.mean(p == y) * 100)


# =================== 2.4 Top predictors for spam ================
# Load the test dataset

mat_data = sio.loadmat('spamTest.mat')
X_test = mat_data['Xtest']
y_test = mat_data['ytest'].ravel()

print('Evaluating the trained Linear SVM on a test set...')
p = clf.predict(X_test)

print('Test Accuracy:', np.mean(p == y_test) * 100)

coef = clf.coef_.ravel()
idx = coef.argsort()[::-1]
vocab_list = getVocablist()

print('Top predictors of spam:')
for i in range(15):
    print("{0:<15s} ({1:f})".format(vocab_list[idx[i]], coef[idx[i]]))


# =================== Part 6: Try Your Own Emails =====================
filename = 'spamSample1.txt'
with open(filename) as f:
    file_contents = f.read().replace('\n', '')

word_indices = processEmail(file_contents)
x = emailFeatures(word_indices)
p = clf.predict(x.T)
print('Processed', filename, '\nSpam Classification:', p)
print('(1 indicates spam, 0 indicates not spam)')
