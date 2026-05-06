def _roc(y_true, y_score, ax=None):
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples]
        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # check dimensions

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=('ROC curve (area = {0:0.2f})'.format(roc_auc)))
    _set_ax_settings(ax)
    return ax