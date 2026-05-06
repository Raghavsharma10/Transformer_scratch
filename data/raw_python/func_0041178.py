def noglobals(fn):
    """ decorator for functions that dont get access to globals """
    return type(fn)(
        getattr(fn, 'func_code', getattr(fn, '__code__')),
        {'__builtins__': builtins},
        getattr(fn, 'func_name', getattr(fn, '__name__')),
        getattr(fn, 'func_defaults', getattr(fn, '__defaults__')),
        getattr(fn, 'func_closure', getattr(fn, '__closure__'))
    )