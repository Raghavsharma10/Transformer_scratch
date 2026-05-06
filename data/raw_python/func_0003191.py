def confusion_matrix(y_true, y_pred, target_names=None, normalize=False,
                     cmap=None, ax=None):
    """
    Plot confustion matrix.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions).
    target_names : list
        List containing the names of the target classes. List must be in order
        e.g. ``['Label for class 0', 'Label for class 1']``. If ``None``,
        generic labels will be generated e.g. ``['Class 0', 'Class 1']``
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    normalize : bool
        Normalize the confusion matrix
    cmap : matplotlib Colormap
        If ``None`` uses a modified version of matplotlib's OrRd colormap.


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/confusion_matrix.py

    """
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion "
                         "matrix")

    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)

    if target_names and (expected_len != len(target_names)):
        raise ValueError(('Data cointains {} different values, but target'
                         ' names contains {} values.'.format(expected_len,
                                                             len(target_names)
                                                             )))

    # if the user didn't pass target_names, create generic ones
    if not target_names:
        values = list(values)
        values.sort()
        target_names = ['Class {}'.format(v) for v in values]

    cm = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    if ax is None:
        ax = plt.gca()

    # this (y, x) may sound counterintuitive. The reason is that
    # in a matrix cell (i, j) is in row=i and col=j, translating that
    # to an x, y plane (which matplotlib uses to plot), we need to use
    # i as the y coordinate (how many steps down) and j as the x coordinate
    # how many steps to the right.
    for (y, x), v in np.ndenumerate(cm):
        try:
            label = '{:.2}'.format(v)
        except:
            label = v
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center')

    if cmap is None:
        cmap = default_heatmap()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    title = 'Confusion matrix'
    if normalize:
        title += ' (normalized)'
    ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax