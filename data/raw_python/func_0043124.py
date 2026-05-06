def floor(value, mod=1):
    """
    x == floor(x, a) + mod(x, a)  FOR ALL a, x
    RETURN None WHEN GIVEN INVALID ARGUMENTS
    """
    if value == None:
        return None
    elif mod <= 0:
        return None
    elif mod == 1:
        return int(math_floor(value))
    elif is_integer(mod):
        return int(math_floor(value / mod)) * mod
    else:
        return math_floor(value / mod) * mod