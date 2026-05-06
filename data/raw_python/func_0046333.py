def serialize(struct, format, target=None, encoding='utf-8'):
    """Serialize given structure and return it as encoded string or write it to file-like object.

    Args:
        struct: structure (dict or list) with unicode members to serialize; note that list
            can only be serialized to json
        format: specify markup format to serialize structure as
        target: binary-opened file-like object to serialize to; if None (default),
            the result will be returned instead of writing to `target`
        encoding: encoding to use when serializing, defaults to utf-8
    Returns:
        bytestring with serialized structure if `target` is None; return value of
        `target.write` otherwise
    Raises:
        AnyMarkupError if a problem occurs while serializing
    """
    # raise if "unicode-opened"
    if hasattr(target, 'encoding') and target.encoding:
        raise AnyMarkupError('Input file must be opened in binary mode')

    fname = None
    if hasattr(target, 'name'):
        fname = target.name

    fmt = _get_format(format, fname)

    try:
        serialized = _do_serialize(struct, fmt, encoding)
        if target is None:
            return serialized
        else:
            return target.write(serialized)
    except Exception as e:
        raise AnyMarkupError(e, traceback.format_exc())