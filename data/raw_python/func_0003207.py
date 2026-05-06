def _roc_multi(y_true, y_score, ax=None):
    """
    Plot ROC curve for multi classification.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples, n_classes]
        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    if ax is None:
        ax = plt.gca()

    ax.plot(fpr, tpr, label=('micro-average ROC curve (area = {0:0.2f})'
                             .format(roc_auc)))
    _set_ax_settings(ax)
    return ax