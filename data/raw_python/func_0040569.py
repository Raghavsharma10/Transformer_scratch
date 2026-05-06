def _normal_function(x, mean, variance):
    """
    Find a value in the cumulative distribution function of a normal curve.

    See https://en.wikipedia.org/wiki/Normal_distribution

    Args:
        x (float): Value to feed into the normal function
        mean (float): Mean of the normal function
        variance (float): Variance of the normal function

    Returns: float

    Example:
        >>> round(_normal_function(0, 0, 5), 4)
        0.1784
    """
    e_power = -1 * (((x - mean) ** 2) / (2 * variance))
    return (1 / math.sqrt(2 * variance * math.pi)) * (math.e ** e_power)