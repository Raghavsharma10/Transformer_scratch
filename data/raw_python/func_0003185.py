def _precision_recall(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    if ax is None:
        ax = plt.gca()

    ax.plot(recall, precision, label=('Precision-Recall curve: AUC={0:0.2f}'
                                      .format(average_precision)))
    _set_ax_settings(ax)
    return ax