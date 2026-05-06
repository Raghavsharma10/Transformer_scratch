def _clamp_value(value, minimum, maximum):
    """
    Clamp a value to fit between a minimum and a maximum.

    * If ``value`` is between ``minimum`` and ``maximum``, return ``value``
    * If ``value`` is below ``minimum``, return ``minimum``
    * If ``value is above ``maximum``, return ``maximum``

    Args:
        value (float or int): The number to clamp
        minimum (float or int): The lowest allowed return value
        maximum (float or int): The highest allowed return value

    Returns:
        float or int: the clamped value

    Raises:
        ValueError: if maximum < minimum

    Example:
        >>> _clamp_value(3, 5, 10)
        5
        >>> _clamp_value(11, 5, 10)
        10
        >>> _clamp_value(8, 5, 10)
        8
    """
    if maximum < minimum:
        raise ValueError
    if value < minimum:
        return minimum
    elif value > maximum:
        return maximum
    else:
        return value