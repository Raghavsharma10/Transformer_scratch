def wraps(
    wrapped, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES
):
    """Cython-compatible functools.wraps implementation."""
    if not is_cython_function(wrapped):
        return functools.wraps(wrapped, assigned, updated)
    else:
        return lambda wrapper: wrapper