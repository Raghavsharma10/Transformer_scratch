def fingerprint_helper(egg, permute=False, n_perms=1000,
                       match='exact', distance='euclidean', features=None):
    """
    Computes clustering along a set of feature dimensions

    Parameters
    ----------
    egg : quail.Egg
        Data to analyze

    dist_funcs : dict
        Dictionary of distance functions for feature clustering analyses

    Returns
    ----------
    probabilities : Numpy array
      Each number represents clustering along a different feature dimension

    """

    if features is None:
        features = egg.dist_funcs.keys()

    inds = egg.pres.index.tolist()
    slices = [egg.crack(subjects=[i], lists=[j]) for i, j in inds]

    weights = _get_weights(slices, features, distdict, permute, n_perms, match,
                            distance)
    return np.nanmean(weights, axis=0)