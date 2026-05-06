def _do_parse(inp, fmt, encoding, force_types):
    """Actually parse input.

    Args:
        inp: bytes yielding file-like object
        fmt: format to use for parsing
        encoding: encoding of `inp`
        force_types:
            if `True`, integers, floats, booleans and none/null
                are recognized and returned as proper types instead of strings;
            if `False`, everything is converted to strings
            if `None`, backend return value is used
    Returns:
        parsed `inp` (dict or list) containing unicode values
    Raises:
        various sorts of errors raised by used libraries while parsing
    """
    res = {}
    _check_lib_installed(fmt, 'parse')

    if fmt == 'ini':
        cfg = configobj.ConfigObj(inp, encoding=encoding)
        res = cfg.dict()
    elif fmt == 'json':
        if six.PY3:
            # python 3 json only reads from unicode objects
            inp = io.TextIOWrapper(inp, encoding=encoding)
        res = json.load(inp, encoding=encoding)
    elif fmt == 'json5':
        if six.PY3:
            inp = io.TextIOWrapper(inp, encoding=encoding)
        res = json5.load(inp, encoding=encoding)
    elif fmt == 'toml':
        if not _is_utf8(encoding):
            raise AnyMarkupError('toml is always utf-8 encoded according to specification')
        if six.PY3:
            # python 3 toml prefers unicode objects
            inp = io.TextIOWrapper(inp, encoding=encoding)
        res = toml.load(inp)
    elif fmt == 'xml':
        res = xmltodict.parse(inp, encoding=encoding)
    elif fmt == 'yaml':
        # guesses encoding by its own, there seems to be no way to pass
        #  it explicitly
        res = yaml.safe_load(inp)
    else:
        raise  # unknown format

    # make sure it's all unicode and all int/float values were parsed correctly
    #   the unicode part is here because of yaml on PY2 and also as workaround for
    #   https://github.com/DiffSK/configobj/issues/18#issuecomment-76391689
    return _ensure_proper_types(res, encoding, force_types)