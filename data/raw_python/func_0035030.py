def bind_all(mc, builtin_only=False, stoplist=[],  verbose=False):
    """Recursively apply constant binding to functions in a module or class.

    Use as the last line of the module (after everything is defined, but
    before test code).  In modules that need modifiable globals, set
    builtin_only to True.

    """
    import types
    try:
        d = vars(mc)
    except TypeError:
        return
    for k, v in d.items():
        if isinstance( v, types.FunctionType ) :
            if verbose :
                print( 'make_constants(', v.__name__, ')' )
            newv = _make_constants(v, builtin_only, stoplist,  verbose)
            setattr(mc, k, newv)
        elif type(v) in ( type, types.ModuleType ):
            bind_all(v, builtin_only, stoplist, verbose)