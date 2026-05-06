def get_env(key: str,
            default: Any = None,
            clean: Callable[[str], Any] = lambda v: v):
    '''
    Retrieves a configuration value from the environment variables.
    The given *key* is uppercased and prefixed by ``"BACKEND_"`` and then
    ``"SORNA_"`` if the former does not exist.

    :param key: The key name.
    :param default: The default value returned when there is no corresponding
        environment variable.
    :param clean: A single-argument function that is applied to the result of lookup
        (in both successes and the default value for failures).
        The default is returning the value as-is.

    :returns: The value processed by the *clean* function.
    '''
    key = key.upper()
    v = os.environ.get('BACKEND_' + key)
    if v is None:
        v = os.environ.get('SORNA_' + key)
    if v is None:
        if default is None:
            raise KeyError(key)
        v = default
    return clean(v)