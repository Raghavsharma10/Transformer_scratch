def mtanh_profile(X, n, x0, delta, alpha, h, b, hyper_deriv=None):
    """Profile used with the mtanh function to fit profiles, suitable for use with :py:class:`MeanFunction`.
    
    Only supports univariate data!
    
    Parameters
    ----------
    X : array, (`M`, 1)
        The points to evaluate at.
    n : array, (1,)
        The order of derivative to compute. Only up to first derivatives are
        supported.
    x0 : float
        Pedestal center
    delta : float
        Pedestal halfwidth
    alpha : float
        Core slope
    h : float
        Pedestal height
    b : float
        Pedestal foot
    hyper_deriv : int or None, optional
        The index of the parameter to take a derivative with respect to.
    """
    X = X[:, 0]
    z = (x0 - X) / delta
    if n[0] == 0:
        if hyper_deriv is not None:
            if hyper_deriv == 0:
                return (h - b) / (2.0 * delta * (scipy.cosh(z))**2) * (
                    1.0 + alpha / 4.0 * (1.0 + 2.0 * z + scipy.exp(2.0 * z))
                )
            elif hyper_deriv == 1:
                return -(h - b) * z / (2.0 * delta * (scipy.cosh(z))**2) * (
                    1.0 + alpha / 4.0 * (1.0 + 2.0 * z + scipy.exp(2.0 * z))
                )
            elif hyper_deriv == 2:
                ez = scipy.exp(z)
                enz = 1.0 / ez
                return (h - b) / 2.0 * z * ez / (ez + enz)
            elif hyper_deriv == 3:
                ez = scipy.exp(z)
                enz = 1.0 / ez
                return  0.5 * (1.0 + ((1.0 + alpha * z) * ez - enz) / (ez + enz))
            elif hyper_deriv == 4:
                ez = scipy.exp(z)
                enz = 1.0 / ez
                return  0.5 * (1.0 - ((1.0 + alpha * z) * ez - enz) / (ez + enz))
            else:
                raise ValueError("Invalid value for hyper_deriv, " + str(hyper_deriv))
        else:
            return (h + b) / 2.0 + (h - b) * mtanh(alpha, z) / 2.0
    elif n[0] == 1:
        if hyper_deriv is not None:
            if hyper_deriv == 0:
                return -(h - b) / (2.0 * delta**2.0 * (scipy.cosh(z))**2.0) * (
                    alpha - (alpha * z + 2) * scipy.tanh(z)
                )
            elif hyper_deriv == 1:
                return (h - b) / (2.0 * delta**2.0 * (scipy.cosh(z))**2.0) * (
                    1.0 + alpha / 4.0 * (1.0 + 2.0 * z + scipy.exp(2.0 * z)) +
                    z * (alpha - (alpha * z + 2) * scipy.tanh(z))
                )
            elif hyper_deriv == 2:
                return -(h - b) / (8.0 * delta * (scipy.cosh(z))**2.0) * (
                    1.0 + 2.0 * z + scipy.exp(2.0 * z)
                )
            elif hyper_deriv == 3:
                return -1.0 / (2.0 * delta * (scipy.cosh(z))**2.0) * (
                    1.0 + alpha / 4.0 * (1.0 + 2.0 * z + scipy.exp(2.0 * z))
                )
            elif hyper_deriv == 4:
                return 1.0 / (2.0 * delta * (scipy.cosh(z))**2.0) * (
                    1.0 + alpha / 4.0 * (1.0 + 2.0 * z + scipy.exp(2.0 * z))
                )
            else:
                raise ValueError("Invalid value for hyper_deriv, " + str(hyper_deriv))
        else:
            return -(h - b) / (2.0 * delta * (scipy.cosh(z))**2) * (
                1 + alpha / 4.0 * (1 + 2 * z + scipy.exp(2 * z))
            )
    else:
        raise NotImplementedError("Derivatives of order greater than 1 are not supported!")