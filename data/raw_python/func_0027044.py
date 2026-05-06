def get_function_call_str(fn, args, kwargs):
    """Converts method call (function and its arguments) to a str(...)-like string."""

    def str_converter(v):
        try:
            return str(v)
        except Exception:
            try:
                return repr(v)
            except Exception:
                return "<n/a str raised>"

    result = get_full_name(fn) + "("
    first = True
    for v in args:
        if first:
            first = False
        else:
            result += ","
        result += str_converter(v)
    for k, v in kwargs.items():
        if first:
            first = False
        else:
            result += ","
        result += str(k) + "=" + str_converter(v)
    result += ")"
    return result