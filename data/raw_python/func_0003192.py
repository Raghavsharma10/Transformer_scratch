def feature_importances(data, top_n=None, feature_names=None, ax=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array-like structure. If data is a scikit-learn model with
    sub-estimators (e.g. RandomForest, AdaBoost) the function will compute the
    standard deviation of each feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature names
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/feature_importances.py

    """
    if data is None:
        raise ValueError('data is needed to plot feature importances. '
                         'When plotting using the evaluator you need to pass '
                         'an estimator ')

    # If no feature_names is provided, assign numbers
    res = compute.feature_importances(data, top_n, feature_names)
    # number of features returned
    n_feats = len(res)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Feature importances")

    try:
        ax.bar(range(n_feats), res.importance, yerr=res.std_, color='red',
               align="center")
    except:
        ax.bar(range(n_feats), res.importance, color='red',
               align="center")

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(res.feature_name)
    ax.set_xlim([-1, n_feats])
    return ax