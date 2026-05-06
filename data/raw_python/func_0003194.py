def feature_importances(data, top_n=None, feature_names=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array-like structure.

    If data is a scikit-learn model with sub-estimators (e.g. RandomForest,
    AdaBoost) the function will compute the standard deviation of each
    feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature_names

    Returns
    -------
    table
        Table object with the data. Columns are
        feature_name, importance (`std_` only included for models with
        sub-estimators)

    """
    if data is None:
        raise ValueError('data is needed to tabulate feature importances. '
                         'When plotting using the evaluator you need to pass '
                         'an estimator ')

    res = compute.feature_importances(data, top_n, feature_names)
    return Table(res, res.dtype.names)