def annotate(*args, **kwargs):
    """Set function annotations (on Python2 and 3)."""
    def decorator(f):
        if not hasattr(f, '__annotations__'):
            f.__annotations__ = kwargs.copy()
        else:
            f.__annotations__.update(kwargs)

        if args:
            if len(args) != 1:
                raise ValueError('annotate supports only a single argument.')
            f.__annotations__['return'] = args[0]
        return f

    return decorator