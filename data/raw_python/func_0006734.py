def list(name, default=None, allow_none=False, fallback=None, separator=','):
    """Get a list of strings or the default.

    The individual list elements are whitespace-stripped.

    Args:
        name: The environment variable name
        default: The default value to use if no environment variable is found
        allow_none: If the return value can be `None` (i.e. optional)
        separator: The list item separator character or pattern
    """
    value = read(name, default, allow_none, fallback=fallback)
    if isinstance(value, builtins.list):
        return value
    elif isinstance(value, builtins.str):
        return _str_to_list(value, separator)
    elif value is None and allow_none:
        return None
    else:
        return [builtins.str(value)]