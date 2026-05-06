def base(number, input_base=10, output_base=10, max_depth=10,
         string=False, recurring=True):
    """
    Converts a number from any base to any another.

    Args:
        number(tuple|str|int): The number to convert.
        input_base(int): The base to convert from (defualt 10).
        output_base(int): The base to convert to (default 10).
        max_depth(int): The maximum number of fractional digits (defult 10).
        string(bool): If True output will be in string representation,
            if False output will be in tuple representation (defult False).
        recurring(bool): Attempt to find repeating digits in the fractional
            part of a number. Repeated digits will be enclosed with "[" and "]"
            (default True).
    Returns:
        A tuple of digits in the specified base:
        (int, int, int, ... , '.' , int, int, int)
        If the string flag is set to True,
        a string representation will be used instead.

    Raises:
        ValueError if a digit value is too high for the input_base.

    Example:
        >>> base((1,9,6,'.',5,1,6), 17, 20)
        (1, 2, 8, '.', 5, 19, 10, 7, 17, 2, 13, 13, 1, 8)
    """
    # Convert number to tuple representation.
    if type(number) == int or type(number) == float:
        number = str(number)
    if type(number) == str:
        number = represent_as_tuple(number)
    # Check that the number is valid for the input base.
    if not check_valid(number, input_base):
        raise ValueError
    # Deal with base-1 special case
    if input_base == 1:
        number = (1,) * number.count(1)
    # Expand any recurring digits.
    number = expand_recurring(number, repeat=5)
    # Convert a fractional number.
    if "." in number:
        radix_point = number.index(".")
        integer_part = number[:radix_point]
        fractional_part = number[radix_point:]
        integer_part = integer_base(integer_part, input_base, output_base)
        fractional_part = fractional_base(fractional_part, input_base,
                                          output_base, max_depth)
        number = integer_part + fractional_part
        number = truncate(number)
    # Convert an integer number.
    else:
        number = integer_base(number, input_base, output_base)
    if recurring:
        number = find_recurring(number, min_repeat=2)
    # Return the converted number as a srring or tuple.
    return represent_as_string(number) if string else number