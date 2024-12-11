import numpy as np
from sklearn.metrics import roc_curve, auc

def roc_auc_value(correct, probabilities):

    correctness = np.array(correct)

    softmax_max = np.max(probabilities, 1)

    fpr, tpr, _ = roc_curve(correctness, softmax_max)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc
