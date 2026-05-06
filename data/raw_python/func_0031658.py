def digits(number, base=10):
    """
    Determines the number of digits of a number in a specific base.

    Args:
        number(int): An integer number represented in base 10.
        base(int): The base to find the number of digits.

    Returns:
        Number of digits when represented in a particular base (integer).

    Examples:
        >>> digits(255)
        3
        >>> digits(255, 16)
        2
        >>> digits(256, 16)
        3
        >>> digits(256, 2)
        9
        >>> digits(0, 678363)
        0
        >>> digits(-1, 678363)
        0
        >>> digits(12345, 10)
        5
    """
    if number < 1:
        return 0
    digits = 0
    n = 1
    while(number >= 1):
        number //= base
        digits += 1
    return digits