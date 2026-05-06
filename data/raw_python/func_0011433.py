def Printf(params, ctxt, scope, stream, coord, interp):
    """Prints format string to stdout

    :params: TODO
    :returns: TODO

    """
    if len(params) == 1:
        if interp._printf:
            sys.stdout.write(PYSTR(params[0]))
        return len(PYSTR(params[0]))

    parts = []
    for part in params[1:]:
        if isinstance(part, pfp.fields.Array) or isinstance(part, pfp.fields.String):
            parts.append(PYSTR(part))
        else:
            parts.append(PYVAL(part))

    to_print = PYSTR(params[0]) % tuple(parts)
    res = len(to_print)

    if interp._printf:
        sys.stdout.write(to_print)
        sys.stdout.flush()
    return res