def get_original_fn(fn):
    """Gets the very original function of a decorated one."""

    fn_type = type(fn)
    if fn_type is classmethod or fn_type is staticmethod:
        return get_original_fn(fn.__func__)
    if hasattr(fn, "original_fn"):
        return fn.original_fn
    if hasattr(fn, "fn"):
        fn.original_fn = get_original_fn(fn.fn)
        return fn.original_fn
    return fn