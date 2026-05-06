def get_function_call_repr(fn, args, kwargs):
    """Converts method call (function and its arguments) to a repr(...)-like string."""

    result = get_full_name(fn) + "("
    first = True
    for v in args:
        if first:
            first = False
        else:
            result += ","
        result += repr(v)
    for k, v in kwargs.items():
        if first:
            first = False
        else:
            result += ","
        result += str(k) + "=" + repr(v)
    result += ")"
    return result