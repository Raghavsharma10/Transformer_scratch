def name_targets(func):
    """
    Wrap a function such that returning ``'a', 'b', 'c', [1, 2, 3]`` transforms
    the value into ``dict(a=1, b=2, c=3)``.

    This is useful in the case where the last parameter is an SCons command.
    """
    def wrap(*a, **kw):
        ret = func(*a, **kw)
        return dict(zip(ret[:-1], ret[-1]))
    return wrap