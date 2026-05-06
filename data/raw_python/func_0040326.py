def see_doc(obj_with_doc):
    """Copy docstring from existing object to the decorated callable."""
    def decorator(fn):
        fn.__doc__ = obj_with_doc.__doc__
        return fn
    return decorator