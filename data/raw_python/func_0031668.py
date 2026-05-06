def check_valid(number, input_base=10):
    """
    Checks if there is an invalid digit in the input number.

    Args:
        number: An number in the following form:
            (int, int, int, ... , '.' , int, int, int)
            (iterable container) containing positive integers of the input base
        input_base(int): The base of the input number.

    Returns:
        bool, True if all digits valid, else False.

    Examples:
        >>> check_valid((1,9,6,'.',5,1,6), 12)
        True
        >>> check_valid((8,1,15,9), 15)
        False
    """
    for n in number:
        if n in (".", "[", "]"):
            continue
        elif n >= input_base:
            if n == 1 and input_base == 1:
                continue
            else:
                return False
    return True