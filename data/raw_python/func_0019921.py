def acosh(x):
    """
    Inverse hyperbolic cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccosh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccosh(x)