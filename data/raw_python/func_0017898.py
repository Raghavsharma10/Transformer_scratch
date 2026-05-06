def get_feature_set_all():
    """
    Return a list of entire features.

    A set of entire features regardless of being used to train a model or
    predict a class.

    Returns
    -------
    feature_names : list
        A list of features' names.
    """

    features = get_feature_set()

    features.append('cusum')
    features.append('eta')
    features.append('n_points')
    features.append('period_SNR')
    features.append('period_log10FAP')
    features.append('period_uncertainty')
    features.append('weighted_mean')
    features.append('weighted_std')

    features.sort()

    return features