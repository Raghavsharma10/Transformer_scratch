def log(x):
    """
    Natural logarithm
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.log(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.log(x)