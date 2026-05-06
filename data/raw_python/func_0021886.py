def round_float(f, digits, rounding=ROUND_HALF_UP):
    """
    Accurate float rounding from http://stackoverflow.com/a/15398691.
    """
    return Decimal(str(f)).quantize(Decimal(10) ** (-1 * digits),
                                    rounding=rounding)