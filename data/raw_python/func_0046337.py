def _ensure_proper_types(struct, encoding, force_types):
    """A convenience function that recursively makes sure the given structure
    contains proper types according to value of `force_types`.

    Args:
        struct: a structure to check and fix
        encoding: encoding to use on found bytestrings
        force_types:
            if `True`, integers, floats, booleans and none/null
                are recognized and returned as proper types instead of strings;
            if `False`, everything is converted to strings
            if `None`, unmodified `struct` is returned
    Returns:
        a fully decoded copy of given structure
    """
    if force_types is None:
        return struct

    # if it's an empty value
    res = None
    if isinstance(struct, (dict, collections.OrderedDict)):
        res = type(struct)()
        for k, v in struct.items():
            res[_ensure_proper_types(k, encoding, force_types)] = \
                _ensure_proper_types(v, encoding, force_types)
    elif isinstance(struct, list):
        res = []
        for i in struct:
            res.append(_ensure_proper_types(i, encoding, force_types))
    elif isinstance(struct, six.binary_type):
        res = struct.decode(encoding)
    elif isinstance(struct, (six.text_type, type(None), type(True), six.integer_types, float)):
        res = struct
    elif isinstance(struct, datetime.datetime):
        # toml can parse datetime natively
        res = struct
    else:
        raise AnyMarkupError('internal error - unexpected type {0} in parsed markup'.
                             format(type(struct)))

    if force_types and isinstance(res, six.text_type):
        res = _recognize_basic_types(res)
    elif not (force_types or
              isinstance(res, (dict, collections.OrderedDict, list, six.text_type))):
        res = six.text_type(res)

    return res