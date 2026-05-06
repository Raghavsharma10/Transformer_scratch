def int_to_str_digit(n):
    """
    Converts a positive integer, to a single string character.
    Where: 9 -> "9", 10 -> "A", 11 -> "B", 12 -> "C", ...etc

    Args:
        n(int): A positve integer number.

    Returns:
        The character representation of the input digit of value n (str).
    """
    # 0 - 9
    if n < 10:
        return str(n)
    # A - Z
    elif n < 36:
        return chr(n + 55)
    # a - z or higher
    else:
        return chr(n + 61)