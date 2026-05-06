def IntGreaterThanZero(n):
    """If *n* is an integer > 0, returns it, otherwise an error."""
    try:
        n = int(n)
    except:
        raise ValueError("%s is not an integer" % n)
    if n <= 0:
        raise ValueError("%d is not > 0" % n)
    else:
        return n