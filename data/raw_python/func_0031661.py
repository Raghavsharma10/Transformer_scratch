def to_base_10_int(n, input_base):
    """
    Converts an integer in any base into it's decimal representation.

    Args:
        n - An integer represented as a tuple of digits in the specified base.
        input_base - the base of the input number.

    Returns:
        integer converted into base 10.

    Example:
        >>> to_base_10_int((8,1), 16)
        129
    """
    return sum(c * input_base ** i for i, c in enumerate(n[::-1]))