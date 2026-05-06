def _mod(value, mod=1):
    """
    RETURN NON-NEGATIVE MODULO
    RETURN None WHEN GIVEN INVALID ARGUMENTS
    """
    if value == None:
        return None
    elif mod <= 0:
        return None
    elif value < 0:
        return (value % mod + mod) % mod
    else:
        return value % mod