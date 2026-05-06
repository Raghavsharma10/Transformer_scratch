def IntGreaterThanOne(n):
    """If *n* is an integer > 1, returns it, otherwise an error."""
    try:
        n = int(n)
    except:
        raise ValueError("%s is not an integer" % n)
    if n <= 1:
        raise ValueError("%d is not > 1" % n)
    else:
        return n