def sampler(X, y):
    '''A basic generator for sampling data.

    Parameters
    ----------
    X : np.ndarray, len=n_samples, ndim=4
        Image data.

    y : np.ndarray, len=n_samples, ndim=2
        One-hot encoded class vectors.

    Yields
    ------
    data : dict
        Single image sample, like {X: np.ndarray, y: np.ndarray}
    '''
    X = np.atleast_2d(X)
    # y's are binary vectors, and should be of shape (10,) after this.
    y = np.atleast_1d(y)

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[i], 'y': y[i]}