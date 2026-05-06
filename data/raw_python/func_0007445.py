def sub(value, arg):
    """Subtract the arg from the value."""
    try:
        nvalue, narg = handle_float_decimal_combinations(
            valid_numeric(value), valid_numeric(arg), '-')
        return nvalue - narg
    except (ValueError, TypeError):
        try:
            return value - arg
        except Exception:
            return ''