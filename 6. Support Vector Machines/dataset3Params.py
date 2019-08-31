from sklearn import svm

def dataset3Params(X, y, Xval, yval):
    #DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
    #   sigma. You should complete this function to return the optimal C and
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    #C = 1;
    #sigma = 0.3;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval);
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(double(predictions ~= yval))
    #

    C_cands = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_cands = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_score = -1
    C_pick = -1
    gamma_pick = -1
    for C in C_cands:
        for gamma in gamma_cands:
            clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            clf.fit(X, y)
            score = clf.score(Xval, yval)
            if score > max_score:
                max_score = score
                C_pick = C
                gamma_pick = gamma

    return C_pick, gamma_pick
