def hypot(x, y):
    """
    Calculate the hypotenuse given two "legs" of a right triangle
    """
    if isinstance(x, UncertainFunction) or isinstance(x, UncertainFunction):
        ufx = to_uncertain_func(x)
        ufy = to_uncertain_func(y)
        mcpts = np.hypot(ufx._mcpts, ufy._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.hypot(x, y)