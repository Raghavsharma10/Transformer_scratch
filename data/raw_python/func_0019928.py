def cosh(x):
    """
    Hyperbolic cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cosh(x)