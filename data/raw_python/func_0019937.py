def log1p(x):
    """
    Natural logarithm of (1 + x)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log1p(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log1p(x)