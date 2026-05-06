def _parse_arg(func, variables, arg_name, anno):
    """Parse an argument's annotation."""
    if isinstance(anno, str):
        var = variables[anno]
        return var, var.read_latest
    elif (isinstance(anno, list) and len(anno) == 1 and
          isinstance(anno[0], str)):
        var = variables[anno[0]]
        return var, var.read_all
    # For now, be very strict about annotation format (e.g.,
    # allow list but not tuple) because we might want to use
    # tuple for other meanings in the future.
    raise StartupError(
        'cannot parse annotation %r of parameter %r for %r' %
        (anno, arg_name, func))