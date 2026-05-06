def parse(inp, format=None, encoding='utf-8', force_types=True):
    """Parse input from file-like object, unicode string or byte string.

    Args:
        inp: file-like object, unicode string or byte string with the markup
        format: explicitly override the guessed `inp` markup format
        encoding: `inp` encoding, defaults to utf-8
        force_types:
            if `True`, integers, floats, booleans and none/null
                are recognized and returned as proper types instead of strings;
            if `False`, everything is converted to strings
            if `None`, backend return value is used
    Returns:
        parsed input (dict or list) containing unicode values
    Raises:
        AnyMarkupError if a problem occurs while parsing or inp
    """
    proper_inp = inp
    if hasattr(inp, 'read'):
        proper_inp = inp.read()
    # if proper_inp is unicode, encode it
    if isinstance(proper_inp, six.text_type):
        proper_inp = proper_inp.encode(encoding)

    # try to guess markup type
    fname = None
    if hasattr(inp, 'name'):
        fname = inp.name
    fmt = _get_format(format, fname, proper_inp)

    # make it look like file-like bytes-yielding object
    proper_inp = six.BytesIO(proper_inp)

    try:
        res = _do_parse(proper_inp, fmt, encoding, force_types)
    except Exception as e:
        # I wish there was only Python 3 and I could just use "raise ... from e"
        raise AnyMarkupError(e, traceback.format_exc())
    if res is None:
        res = {}

    return res