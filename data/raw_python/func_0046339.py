def _get_format(format, fname, inp=None):
    """Try to guess markup format of given input.

    Args:
        format: explicit format override to use
        fname: name of file, if a file was used to read `inp`
        inp: optional bytestring to guess format of (can be None, if markup
            format is to be guessed only from `format` and `fname`)
    Returns:
        guessed format (a key of fmt_to_exts dict)
    Raises:
        AnyMarkupError if explicit format override has unsupported value
            or if it's impossible to guess the format
    """
    fmt = None
    err = True

    if format is not None:
        if format in fmt_to_exts:
            fmt = format
            err = False
    elif fname:
        # get file extension without leading dot
        file_ext = os.path.splitext(fname)[1][len(os.path.extsep):]
        for fmt_name, exts in fmt_to_exts.items():
            if file_ext in exts:
                fmt = fmt_name
                err = False

    if fmt is None:
        if inp is not None:
            fmt = _guess_fmt_from_bytes(inp)
            err = False

    if err:
        err_string = 'Failed to guess markup format based on: '
        what = []
        for k, v in {format: 'specified format argument',
                     fname: 'filename', inp: 'input string'}.items():
            if k:
                what.append(v)
        if not what:
            what.append('nothing to guess format from!')
        err_string += ', '.join(what)
        raise AnyMarkupError(err_string)

    return fmt