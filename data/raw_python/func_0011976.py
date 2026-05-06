def _to_scientific_tuple(number):
    r"""
    Return mantissa and exponent of a number expressed in scientific notation.

    Full precision is maintained if the number is represented as a string.

    :param number: Number
    :type  number: integer, float or string

    :rtype: Tuple whose first item is the mantissa (*string*) and the second
            item is the exponent (*integer*) of the number when expressed in
            scientific notation

    :raises: RuntimeError (Argument \`number\` is not valid)
    """
    # pylint: disable=W0632
    if isinstance(number, bool) or (not isinstance(number, (int, float, str))):
        raise RuntimeError("Argument `number` is not valid")
    convert = not isinstance(number, str)
    # Detect zero and return, simplifies subsequent algorithm
    if (convert and (not number)) or (
        (not convert) and (not number.strip("0").strip("."))
    ):
        return ("0", 0)
    # Break down number into its components, use Decimal type to
    # preserve resolution:
    # sign  : 0 -> +, 1 -> -
    # digits: tuple with digits of number
    # exp   : exponent that gives null fractional part
    sign, digits, exp = Decimal(str(number) if convert else number).as_tuple()
    mant = (
        "{sign}{itg}.{frac}".format(
            sign="-" if sign else "",
            itg=digits[0],
            frac="".join(str(item) for item in digits[1:]),
        )
        .rstrip("0")
        .rstrip(".")
    )
    exp += len(digits) - 1
    return (mant, exp)