def ceil(x):
    """
    Ceiling function (round towards positive infinity)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.ceil(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.ceil(x)