def shorten_type(typ):
    """ Shorten a type. E.g. drops 'System.' """
    offset = 0
    for prefix in SHORTEN_TYPE_PREFIXES:
        if typ.startswith(prefix):
            if len(prefix) > offset:
                offset = len(prefix)
    return typ[offset:]