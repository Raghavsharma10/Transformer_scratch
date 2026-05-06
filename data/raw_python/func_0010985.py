def safe_power(a, b):
    """
    Same power of a ^ b
    :param a: Number a
    :param b: Number b
    :return: a ^ b
    """
    if abs(a) > MAX_POWER or abs(b) > MAX_POWER:
        raise ValueError('Number too high!')
    return a ** b