def FloatBetweenZeroAndOne(x):
    """Returns *x* only if *0 <= x <= 1*, otherwise raises error."""
    x = float(x)
    if 0 <= x <= 1:
        return x
    else:
        raise ValueError("{0} not a float between 0 and 1.".format(x))