def decimal_to_ratio(d):
    """Convert Decimal d to exact integer ratio (numerator, denominator).
    """
    sign, digits, exp = d.as_tuple()
    if exp in ('F', 'n', 'N'):  # INF, NAN, sNAN
        assert not d.is_finite()
        raise ValueError
    num = 0
    for digit in digits:
        num = num * 10 + digit
    if sign:
        num = -num
    den = 10 ** -exp
    return (num, den)