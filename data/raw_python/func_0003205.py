def roc(y_true, y_score, ax=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
              classification or [n_samples, n_classes] for multiclass

        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Notes
    -----
    It is assumed that the y_score parameter columns are in order. For example,
    if ``y_true = [2, 2, 1, 0, 0, 1, 2]``, then the first column in y_score
    must countain the scores for class 0, second column for class 1 and so on.


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/roc.py

    """
    if any((val is None for val in (y_true, y_score))):
        raise ValueError("y_true and y_score are needed to plot ROC")

    if ax is None:
        ax = plt.gca()

    # get the number of classes based on the shape of y_score
    y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
    if y_score_is_vector:
        n_classes = 2
    else:
        _, n_classes = y_score.shape

    # check data shape?

    if n_classes > 2:
        # convert y_true to binary format
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        _roc_multi(y_true_bin, y_score, ax=ax)
        for i in range(n_classes):
            _roc(y_true_bin[:, i], y_score[:, i], ax=ax)
    else:
        if y_score_is_vector:
            _roc(y_true, y_score, ax)
        else:
            _roc(y_true, y_score[:, 1], ax)

    # raise error if n_classes = 1?
    return ax