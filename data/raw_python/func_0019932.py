def fabs(x):
    """
    Absolute value function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.fabs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.fabs(x)