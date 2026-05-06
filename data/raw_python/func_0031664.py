def str_digit_to_int(chr):
    """
    Converts a string character to a decimal number.
    Where "A"->10, "B"->11, "C"->12, ...etc

    Args:
        chr(str): A single character in the form of a string.

    Returns:
        The integer value of the input string digit.
    """
    # 0 - 9
    if chr in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        n = int(chr)
    else:
        n = ord(chr)
        # A - Z
        if n < 91:
            n -= 55
        # a - z or higher
        else:
            n -= 61
    return n