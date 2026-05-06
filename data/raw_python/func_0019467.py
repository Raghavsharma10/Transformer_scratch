def pr_auc(fg_vals, bg_vals):
    """
    Computes the Precision-Recall Area Under Curve (PR AUC)

    Parameters
    ----------
    fg_vals : array_like
        list of values for positive set

    bg_vals : array_like
        list of values for negative set
    
    Returns
    -------
    score : float
        PR AUC score
    """
    # Create y_labels
    y_true, y_score = values_to_labels(fg_vals, bg_vals)
    
    return average_precision_score(y_true, y_score)