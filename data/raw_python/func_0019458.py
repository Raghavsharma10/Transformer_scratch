def values_to_labels(fg_vals, bg_vals):
    """
    Convert two arrays of values to an array of labels and an array of scores.

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.

    Returns
    -------
    y_true : array
        Labels.
    y_score : array
        Values.
    """ 
    y_true = np.hstack((np.ones(len(fg_vals)), np.zeros(len(bg_vals))))
    y_score = np.hstack((fg_vals, bg_vals))
    
    return y_true, y_score