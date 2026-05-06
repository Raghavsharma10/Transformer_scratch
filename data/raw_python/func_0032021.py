def exact_ratio(x):
    """Convert Real number x exactly to (numerator, denominator) pair.

    x is expected to be an int, Fraction, Decimal or float.
    """
    try:
        try:
            # int, Fraction
            return x.numerator, x.denominator
        except AttributeError:
            # float
            try:
                return x.as_integer_ratio()
            except AttributeError:
                # Decimal
                try:
                    return decimal_to_ratio(x)
                except AttributeError:
                    msg = "can't convert type '{}' to numerator/denominator"
                    raise TypeError(msg.format(type(x).__name__))
    except (OverflowError, ValueError):
        # INF or NAN
        return (x, None)