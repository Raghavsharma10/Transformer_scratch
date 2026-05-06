def parse_file(path, format=None, encoding='utf-8', force_types=True):
    """A convenience wrapper of parse, which accepts path of file to parse.

    Args:
        path: path to file to parse
        format: explicitly override the guessed `inp` markup format
        encoding: file encoding, defaults to utf-8
        force_types:
            if `True`, integers, floats, booleans and none/null
                are recognized and returned as proper types instead of strings;
            if `False`, everything is converted to strings
            if `None`, backend return value is used
    Returns:
        parsed `inp` (dict or list) containing unicode values
    Raises:
        AnyMarkupError if a problem occurs while parsing
    """
    try:
        with open(path, 'rb') as f:
            return parse(f, format, encoding, force_types)
    except EnvironmentError as e:
        raise AnyMarkupError(e, traceback.format_exc())