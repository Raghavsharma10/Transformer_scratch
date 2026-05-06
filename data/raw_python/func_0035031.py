def make_constants(builtin_only=False, stoplist=[], verbose=False):
    """
    Return a decorator for optimizing global references.
    Verify that the first argument is a function.
    """
    if type(builtin_only) == type(make_constants):
        raise ValueError("The make_constants decorator must have arguments.")
    return lambda f: _make_constants(f, builtin_only, stoplist, verbose)