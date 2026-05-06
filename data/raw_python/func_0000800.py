def _errcheck(result, func, arguments):
    """
    Error checker for functions returning an integer indicating
    success (0) / failure (1).

    Raises a XdoException in case of error, otherwise just
    returns ``None`` (returning the original code, 0, would be
    useless anyways..)
    """

    if result != 0:
        raise XdoException(
            'Function {0} returned error code {1}'
            .format(func.__name__, result))
    return None