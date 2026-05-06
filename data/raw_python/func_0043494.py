def autocorrelation(X, lag=1):
    """
    Computes the autocorrelation of *X* with the given *lag*.
    Autocorrelation is simply
    autocovariance(X) / covariance(X-mean, X-mean),
    where autocovariance is simply
    covariance((X-mean)[:-lag], (X-mean)[lag:]).

    See `link <https://en.wikipedia.org/wiki/Autocorrelation>`_ for details.

    **Parameters**

    X : array-like, shape = [n_samples]

    lag : int, optional
        Index difference between points being compared (default 1).
    """
    differences = X - X.mean()
    products = differences * concatenate((differences[lag:],
                                          differences[:lag]))

    return products.sum() / (differences**2).sum()