def roc_auc(fg_vals, bg_vals):
    """
    Computes the ROC Area Under Curve (ROC AUC)

    Parameters
    ----------
    fg_vals : array_like
        list of values for positive set

    bg_vals : array_like
        list of values for negative set
    
    Returns
    -------
    score : float
        ROC AUC score
    """
    # Create y_labels
    y_true, y_score = values_to_labels(fg_vals, bg_vals)
    
    return roc_auc_score(y_true, y_score)