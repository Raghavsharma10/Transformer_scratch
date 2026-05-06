def constant(X, n, mu, hyper_deriv=None):
    """Function implementing a constant mean suitable for use with :py:class:`MeanFunction`.
    """
    if (n == 0).all():
        if hyper_deriv is not None:
            return scipy.ones(X.shape[0])
        else:
            return mu * scipy.ones(X.shape[0])
    else:
        return scipy.zeros(X.shape[0])