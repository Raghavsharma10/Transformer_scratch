def positional_args(arguments):
    """"Generator for position arguments.

    Example
    -------
    >>> list(positional_args(["arg1", "arg2", "--kwarg"]))
    ["arg1", "arg2"]
    >>> list(positional_args(["--", "arg1", "--kwarg"]))
    ["arg1", "kwarg"]

    """
    # TODO this behaviour probably isn't quite right.
    if arguments and arguments[0] == '--':
        for a in arguments[1:]:
            yield a
    else:
        for a in arguments:
            if a.startswith('-'):
                break
            yield a