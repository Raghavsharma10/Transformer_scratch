def floor(x):
    """
    Floor function (round towards negative infinity)
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.floor(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.floor(x)