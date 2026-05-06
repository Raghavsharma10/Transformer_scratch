def value_type(value):
    """Classify `value` of bold, color, and underline keys.

    Parameters
    ----------
    value : style value

    Returns
    -------
    str, {"simple", "lookup", "re_lookup", "interval"}
    """
    try:
        keys = list(value.keys())
    except AttributeError:
        return "simple"
    if keys in [["lookup"], ["re_lookup"], ["interval"]]:
        return keys[0]
    raise ValueError("Type of `value` could not be determined")