def ceiling(value, mod=1):
    """
    RETURN SMALLEST INTEGER GREATER THAN value
    """
    if value == None:
        return None
    mod = int(mod)

    v = int(math_floor(value + mod))
    return v - (v % mod)