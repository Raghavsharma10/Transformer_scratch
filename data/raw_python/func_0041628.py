def check_arg_types(funcname, *args):
    """Raise TypeError if not all items of `args` are same string type."""
    hasstr = hasbytes = False
    for arg in args:
        if isinstance(arg, str):
            hasstr = True
        elif isinstance(arg, bytes):
            hasbytes = True
        else:
            raise TypeError('{0}() argument must be str or bytes, not {1}'
                            .format(funcname, arg.__class__.__name__))
    if hasstr and hasbytes:
        raise TypeError("Can't mix strings and bytes in path components")