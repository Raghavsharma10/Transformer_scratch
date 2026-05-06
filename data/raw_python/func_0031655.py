def represent_as_tuple(string):
    """
    Represent a number-string in the form of a tuple of digits.
    "868.0F" -> (8, 6, 8, '.', 0, 15)

    Args:
        string - Number represented as a string of digits.
    Returns:
        Number represented as an iterable container of digits

    >>> represent_as_tuple('868.0F')
    (8, 6, 8, '.', 0, 15)
    """
    keep = (".", "[", "]")
    return tuple(str_digit_to_int(c) if c not in keep else c for c in string)