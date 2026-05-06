def _parseIntegerArgument(args, key, defaultValue):
    """
    Attempts to parse the specified key in the specified argument
    dictionary into an integer. If the argument cannot be parsed,
    raises a BadRequestIntegerException. If the key is not present,
    return the specified default value.
    """
    ret = defaultValue
    try:
        if key in args:
            try:
                ret = int(args[key])
            except ValueError:
                raise exceptions.BadRequestIntegerException(key, args[key])
    except TypeError:
        raise Exception((key, args))
    return ret