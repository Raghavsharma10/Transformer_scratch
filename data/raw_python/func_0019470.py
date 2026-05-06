def roc_values(fg_vals, bg_vals):
    """
    Return fpr (x) and tpr (y) of the ROC curve.

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.

    Returns
    -------
    fpr : array
        False positive rate.
    tpr : array
        True positive rate.
    """
    if len(fg_vals) == 0:
        return 0
    
    y_true, y_score = values_to_labels(fg_vals, bg_vals)
    
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
     
    return fpr, tpr