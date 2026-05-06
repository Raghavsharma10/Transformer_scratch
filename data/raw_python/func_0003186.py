def _precision_recall_multi(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples, n_classes]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # Compute micro-average ROC curve and ROC area
    precision, recall, _ = precision_recall_curve(y_true.ravel(),
                                                  y_score.ravel())

    avg_prec = average_precision_score(y_true, y_score, average="micro")

    if ax is None:
        ax = plt.gca()

    ax.plot(recall, precision,
            label=('micro-average Precision-recall curve (area = {0:0.2f})'
                   .format(avg_prec)))
    _set_ax_settings(ax)
    return ax