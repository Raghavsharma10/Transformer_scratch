def digit(decimal, digit, input_base=10):
    """
    Find the value of an integer at a specific digit when represented in a
    particular base.

    Args:
        decimal(int): A number represented in base 10 (positive integer).
        digit(int): The digit to find where zero is the first, lowest, digit.
        base(int): The base to use (default 10).

    Returns:
        The value at specified digit in the input decimal.
        This output value is represented as a base 10 integer.

    Examples:
        >>> digit(201, 0)
        1
        >>> digit(201, 1)
        0
        >>> digit(201, 2)
        2
        >>> tuple(digit(253, i, 2) for i in range(8))
        (1, 0, 1, 1, 1, 1, 1, 1)

        # Find the lowest digit of a large hexidecimal number
        >>> digit(123456789123456789, 0, 16)
        5
    """
    if decimal == 0:
        return 0
    if digit != 0:
        return (decimal // (input_base ** digit)) % input_base
    else:
        return decimal % input_base