def getfield(f):
    """convert values from cgi.Field objects to plain values."""
    if isinstance(f, list):
        return [getfield(x) for x in f]
    else:
        return f.value