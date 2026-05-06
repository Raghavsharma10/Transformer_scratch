def incver(o, prop_names):
    """Increment the version numbers of a set of properties and return a new object"""
    from ambry.identity import ObjectNumber

    d = {}

    for p in o.__mapper__.attrs:
        v = getattr(o, p.key)
        if v is None:
            d[p.key] = None
        elif p.key in prop_names:
            d[p.key] = str(ObjectNumber.increment(v))
        else:
            if not hasattr(v, '__mapper__'): # Only copy values, never objects
                d[p.key] = v

    return o.__class__(**d)