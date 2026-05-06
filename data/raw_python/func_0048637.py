def _exec(object, globals, locals):
    r"""
    >>> d = {}
    >>> exec('a = 0', d, d)
    >>> d['a']
    0
    """
    if sys.version_info < (3,):
        exec('exec object in globals, locals')
    else:
        exec(object, globals, locals)