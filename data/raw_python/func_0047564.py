def memoize(f):
    """Cache value returned by the function."""
    @wraps(f)
    def w(*args, **kw):
        memoize.mem[f] = v = f(*args, **kw)
        return v
    return w