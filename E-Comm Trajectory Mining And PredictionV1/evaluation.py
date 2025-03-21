import numpy as np

def evaluate_predictions(true_labels, predicted_labels):
    """
    Compute evaluation metrics: P@1, P@6, MRR.

    :param true_labels: List of actual next steps
    :param predicted_labels: List of predicted next steps
    :return: P@1, P@6, MRR
    """
    p1, p6, mrr = 0, 0, 0
    for true, predicted in zip(true_labels, predicted_labels):
        if true == predicted[0]: 
            p1 += 1
        if true in predicted[:6]: 
            p6 += 1
        if true in predicted: 
            mrr += 1 / (predicted.index(true) + 1)
    
    p1 /= len(true_labels)
    p6 /= len(true_labels)
    mrr /= len(true_labels)
    
    return p1, p6, mrr
