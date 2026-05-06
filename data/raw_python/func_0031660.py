def from_base_10_int(decimal, output_base=10):
    """
    Converts a decimal integer to a specific base.

    Args:
        decimal(int) A base 10 number.
        output_base(int) base to convert to.

    Returns:
        A tuple of digits in the specified base.

    Examples:
        >>> from_base_10_int(255)
        (2, 5, 5)
        >>> from_base_10_int(255, 16)
        (15, 15)
        >>> from_base_10_int(9988664439, 8)
        (1, 1, 2, 3, 2, 7, 5, 6, 6, 1, 6, 7)
        >>> from_base_10_int(0, 17)
        (0,)
    """
    if decimal <= 0:
        return (0,)
    if output_base == 1:
        return (1,) * decimal
    length = digits(decimal, output_base)
    converted = tuple(digit(decimal, i, output_base) for i in range(length))
    return converted[::-1]