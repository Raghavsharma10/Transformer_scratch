def read(name, default=None, allow_none=False, fallback=None):
    """Read the raw env value.

    Read the raw environment variable or use the default. If the value is not
    found and no default is set throw an exception.

    Args:
        name: The environment variable name
        default: The default value to use if no environment variable is found
        allow_none: If the return value can be `None` (i.e. optional)
        fallback: A list of fallback env variables to try and read if the primary environment
                  variable is unavailable.
    """
    raw_value = environ.get(name)
    if raw_value is None and fallback is not None:
        if not isinstance(fallback, builtins.list) and not isinstance(fallback, builtins.tuple):
            fallback = [fallback]

        for fall in fallback:
            raw_value = environ.get(fall)
            if raw_value is not None:
                break

    if raw_value or raw_value == '':
        return raw_value
    elif default is not None or allow_none:
        return default
    else:
        raise KeyError('Set the "{0}" environment variable'.format(name))