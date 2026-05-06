def linear(X, n, *args, **kwargs):
    """Linear mean function of arbitrary dimension, suitable for use with :py:class:`MeanFunction`.
    
    The form is :math:`m_0 * X[:, 0] + m_1 * X[:, 1] + \dots + b`.
    
    Parameters
    ----------
    X : array, (`M`, `D`)
        The points to evaluate the model at.
    n : array of non-negative int, (`D`)
        The derivative order to take, specified as an integer order for each
        dimension in `X`.
    *args : num_dim+1 floats
        The slopes for each dimension, plus the constant term. Must be of the
        form `m0, m1, ..., b`.
    """
    hyper_deriv = kwargs.pop('hyper_deriv', None)
    m = scipy.asarray(args[:-1])
    b = args[-1]
    if sum(n) > 1:
        return scipy.zeros(X.shape[0])
    elif sum(n) == 0:
        if hyper_deriv is not None:
            if hyper_deriv < len(m):
                return X[:, hyper_deriv]
            elif hyper_deriv == len(m):
                return scipy.ones(X.shape[0])
            else:
                raise ValueError("Invalid value for hyper_deriv, " + str(hyper_deriv))
        else:
            return (m * X).sum(axis=1) + b
    else:
        # sum(n) == 1:
        if hyper_deriv is not None:
            if n[hyper_deriv] == 1:
                return scipy.ones(X.shape[0])
            else:
                return scipy.zeros(X.shape[0])
        return m[n == 1] * scipy.ones(X.shape[0])