def represent_as_string(iterable):
    """
    Represent a number in the form of a string.
    (8, 6, 8, '.', 0, 15) -> "868.0F"

    Args:
        iterable - Number represented as an iterable container of digits.
    Returns:
        Number represented as a string of digits.

    >>> represent_as_string((8, 6, 8, '.', 0, 15))
    '868.0F'
    """
    keep = (".", "[", "]")
    return "".join(tuple(int_to_str_digit(i) if i not in keep
                   else i for i in iterable))