"""
Utility functions related to evaluation metrics.
"""

def read_conf_matrix(cm, pos_class):
    """ Calculates confusion matrix and returns true and false positives and negatives based on the
    label specified as positive.

    This function assumes that the classes have IDs 0, 1, 2,...
    
    Args:
        y_true: True labels. 1d array or list
        y_pred: Predicted labels. 1d array or list
        classes: List of IDs of class labels. dim=n_classes
        pos_class: ID of label to take as positive class

    Returns:
        cm: confusion matrix of n_labels x n_labels
        tp: true positives
        tn: true negatives
        fp: false positives
        fn: false negatives
    """

    n_classes = cm.shape[0]
    tp = tn = fp = fn = p = n = 0
    for row in range(n_classes):
        for col in range(n_classes):
            if row == col:
                if row == pos_class:
                    tp = cm[row,col]
                else:
                    tn += cm[row,col]
            else:
                if col == pos_class:
                    fp += cm[row,col]
                else:
                    fn += cm[row,col]
            if row == pos_class:        # TODO: it might be possible to optimise this and not have another if statement.
                p += cm[row, col]

    return tp, tn, fp, fn, p, n     # TODO: function doesn't yet calculate 'n'

def precision_score(tp, fp):
    if tp == 0 and fp == 0:
        return 0
    return tp / (tp + fp)

def recall_score(tp, p):
    if tp == 0 and p == 0:
        return 0
    return tp / p

def f1_prec_recall(tp,tn,fp,fn,p,n):
    prec = precision_score(tp, fp)
    rec = recall_score(tp, p)
    if prec == 0 and rec == 0:
        return 0, 0, 0
    f1 = (2 * prec * rec) / (prec + rec)
    return f1, prec, rec

def calc_sens_spec_uar(conf_mat, pos_class):
    tp, tn, fp, fn, p, n = read_conf_matrix(conf_mat, pos_class=pos_class)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    uar = (sens + spec) / 2
    return sens, spec, uar