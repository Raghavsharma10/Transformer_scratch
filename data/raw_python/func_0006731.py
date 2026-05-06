def str(name, default=None, allow_none=False, fallback=None):
    """Get a string based environment value or the default.

    Args:
        name: The environment variable name
        default: The default value to use if no environment variable is found
        allow_none: If the return value can be `None` (i.e. optional)
    """
    value = read(name, default, allow_none, fallback=fallback)
    if value is None and allow_none:
        return None
    else:
        return builtins.str(value).strip()