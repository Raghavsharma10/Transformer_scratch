def pgcd(numa, numb):
    """
    Calculate the greatest common divisor (GCD) of two numbers.

    :param numa: First number
    :type  numa: number

    :param numb: Second number
    :type  numb: number

    :rtype: number

    For example:

        >>> import pmisc, fractions
        >>> pmisc.pgcd(10, 15)
        5
        >>> str(pmisc.pgcd(0.05, 0.02))
        '0.01'
        >>> str(pmisc.pgcd(5/3.0, 2/3.0))[:6]
        '0.3333'
        >>> pmisc.pgcd(
        ...     fractions.Fraction(str(5/3.0)),
        ...     fractions.Fraction(str(2/3.0))
        ... )
        Fraction(1, 3)
        >>> pmisc.pgcd(
        ...     fractions.Fraction(5, 3),
        ...     fractions.Fraction(2, 3)
        ... )
        Fraction(1, 3)
    """
    # Test for integers this way to be valid also for Numpy data types without
    # actually importing (and package depending on) Numpy
    int_args = (int(numa) == numa) and (int(numb) == numb)
    fraction_args = isinstance(numa, Fraction) and isinstance(numb, Fraction)
    # Force conversion for Numpy data types
    if int_args:
        numa, numb = int(numa), int(numb)
    elif not fraction_args:
        numa, numb = float(numa), float(numb)
    # Limit floating numbers to a "sane" fractional part resolution
    if (not int_args) and (not fraction_args):
        numa, numb = (
            Fraction(_no_exp(numa)).limit_denominator(),
            Fraction(_no_exp(numb)).limit_denominator(),
        )
    while numb:
        numa, numb = (
            numb,
            (numa % numb if int_args else (numa % numb).limit_denominator()),
        )
    return int(numa) if int_args else (numa if fraction_args else float(numa))