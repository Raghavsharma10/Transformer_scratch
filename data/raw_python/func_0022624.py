def _class_dispatch(args, kwargs):
    """See 'class_multimethod'."""
    _ = kwargs
    if not args:
        raise ValueError(
            "Multimethods must be passed at least one positional arg.")

    if not isinstance(args[0], type):
        raise TypeError(
            "class_multimethod must be called with a type, not instance.")

    return args[0]