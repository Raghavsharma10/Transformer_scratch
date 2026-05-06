def getargspec(obj):
    """Portable version of inspect.getargspec().

    Necessary because the original is no longer available
    starting from Python 3.6.

    :return: 4-tuple of (argnames, varargname, kwargname, defaults)

    Note that distinction between positional-or-keyword and keyword-only
    parameters will be lost, as the original getargspec() doesn't honor it.
    """
    try:
        return inspect.getargspec(obj)
    except AttributeError:
        pass  # we let a TypeError through

    # translate the signature object back into the 4-tuple
    argnames = []
    varargname, kwargname = None, None
    defaults = []
    for name, param in inspect.signature(obj):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            varargname = name
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            kwargname = name
        else:
            argnames.append(name)
            if param.default is not inspect.Parameter.empty:
                defaults.append(param.default)
    defaults = defaults or None

    return argnames, varargname, kwargname, defaults