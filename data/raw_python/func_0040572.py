def normal_distribution(mean, variance,
                        minimum=None, maximum=None, weight_count=23):
    """
    Return a list of weights approximating a normal distribution.

    Args:
        mean (float): The mean of the distribution
        variance (float): The variance of the distribution
        minimum (float): The minimum outcome possible to
            bound the output distribution to
        maximum (float): The maximum outcome possible to
            bound the output distribution to
        weight_count (int): The number of weights that will
            be used to approximate the distribution

    Returns:
        list: a list of ``(float, float)`` weight tuples
        approximating a normal distribution.

    Raises:
        ValueError: ``if maximum < minimum``
        TypeError: if both ``minimum`` and ``maximum`` are ``None``

    Example:
        >>> weights = normal_distribution(10, 3,
        ...                               minimum=0, maximum=20,
        ...                               weight_count=5)
        >>> rounded_weights = [(round(value, 2), round(strength, 2))
        ...                    for value, strength in weights]
        >>> rounded_weights
        [(1.34, 0.0), (4.8, 0.0), (8.27, 0.14), (11.73, 0.14), (15.2, 0.0)]
    """
    # Pin 0 to +- 5 sigma as bounds, or minimum and maximum
    # if they cross +/- sigma
    standard_deviation = math.sqrt(variance)
    min_x = (standard_deviation * -5) + mean
    max_x = (standard_deviation * 5) + mean
    step = (max_x - min_x) / weight_count
    current_x = min_x
    weights = []
    while current_x < max_x:
        weights.append(
            (current_x, _normal_function(current_x, mean, variance))
        )
        current_x += step
    if minimum is not None or maximum is not None:
        return bound_weights(weights, minimum, maximum)
    else:
        return weights