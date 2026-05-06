def recall_at_fdr(fg_vals, bg_vals, fdr_cutoff=0.1):
    """
    Computes the recall at a specific FDR (default 10%).

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.
    
    fdr : float, optional
        The FDR (between 0.0 and 1.0).
    
    Returns
    -------
    recall : float
        The recall at the specified FDR.
    """
    if len(fg_vals) == 0:
        return 0.0
    
    y_true, y_score = values_to_labels(fg_vals, bg_vals)
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fdr = 1 - precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]