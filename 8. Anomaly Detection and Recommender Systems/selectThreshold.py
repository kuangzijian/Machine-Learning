import numpy as np


def selectThreshold(y_val, p_val):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the F1 score of choosing epsilon as the
    #               threshold and place the value in F1. The code at the
    #               end of the loop will compare the F1 score for this
    #               choice of epsilon and set it to be the best epsilon if
    #               it is better than the current choice of epsilon.
    #               
    # Note: You can use predictions = (pval < epsilon) to get a binary vector
    #       of 0's and 1's of the outlier predictions
    stepSize = (np.max(p_val) - np.min(p_val)) / 1000

    bestEpsilon = 0.0
    bestF1 = 0.0

    for epsilon in np.arange(min(p_val), max(p_val), stepSize):
        predictions = p_val < epsilon
        tp = np.sum(predictions[np.nonzero(y_val == True)])
        fp = np.sum(predictions[np.nonzero(y_val == False)])
        fn = np.sum(y_val[np.nonzero(predictions == False)] == True)
        if tp != 0:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon

    return bestEpsilon, bestF1