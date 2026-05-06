def fractional_base(fractional_part, input_base=10, output_base=10,
                    max_depth=100):
    """
    Convert the fractional part of a number from any base to any base.

    Args:
        fractional_part(iterable container): The fractional part of a number in
            the following form:    ( ".", int, int, int, ...)
        input_base(int): The base to convert from (defualt 10).
        output_base(int): The base to convert to (default 10).
        max_depth(int): The maximum number of decimal digits to output.

    Returns:
        The converted number as a tuple of digits.

    Example:
        >>> fractional_base((".", 6,),10,16,10)
        ('.', 9, 9, 9, 9, 9, 9, 9, 9, 9, 9)
    """
    fractional_part = fractional_part[1:]
    fractional_digits = len(fractional_part)
    numerator = 0
    for i, value in enumerate(fractional_part, 1):
        numerator += value * input_base ** (fractional_digits - i)
    denominator = input_base ** fractional_digits
    i = 1
    digits = []
    while(i < max_depth + 1):
        numerator *= output_base ** i
        digit = numerator // denominator
        numerator -= digit * denominator
        denominator *= output_base ** i
        digits.append(digit)
        i += 1
        greatest_common_divisor = gcd(numerator, denominator)
        numerator //= greatest_common_divisor
        denominator //= greatest_common_divisor
    return (".",) + tuple(digits)