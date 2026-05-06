def _setdef(argdict, name, defaultvalue):
    """Like dict.setdefault but sets the default value also if None is present.

    """
    if not name in argdict or argdict[name] is None:
        argdict[name] = defaultvalue
    return argdict[name]