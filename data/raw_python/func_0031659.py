def integer_fractional_parts(number):
    """
    Returns a tuple of the integer and fractional parts of a number.

    Args:
        number(iterable container): A number in the following form:
            (..., ".", int, int, int, ...)

    Returns:
        (integer_part, fractional_part): tuple.

    Example:
        >>> integer_fractional_parts((1,2,3,".",4,5,6))
        ((1, 2, 3), ('.', 4, 5, 6))
    """
    radix_point = number.index(".")
    integer_part = number[:radix_point]
    fractional_part = number[radix_point:]
    return(integer_part, fractional_part)