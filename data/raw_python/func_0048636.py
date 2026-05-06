def python(source):
    r"""
    >>> python('def add(a, b): return a + b').add(40, 2)
    42
    """
    obj = type('', (object,), {})()
    _exec(source, obj.__dict__, obj.__dict__)
    return obj