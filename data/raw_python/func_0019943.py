def tanh(x):
    """
    Hyperbolic tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tanh(x)