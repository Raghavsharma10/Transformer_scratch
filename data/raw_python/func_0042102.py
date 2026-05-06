def _parse_args(func, variables, annotations=None):
    """Return a list of arguments with the variable it reads.

    NOTE: Multiple arguments may read the same variable.
    """
    arg_read_var = []
    for arg_name, anno in (annotations or func.__annotations__).items():
        if arg_name == 'return':
            continue
        var, read = _parse_arg(func, variables, arg_name, anno)
        arg = Argument(name=arg_name, read=read)
        arg_read_var.append((arg, var))
    return arg_read_var