def serialize_file(struct, path, format=None, encoding='utf-8'):
    """A convenience wrapper of serialize, which accepts path of file to serialize to.

    Args:
        struct: structure (dict or list) with unicode members to serialize; note that list
            can only be serialized to json
        path: path of the file to serialize to
        format: override markup format to serialize structure as (taken from filename
            by default)
        encoding: encoding to use when serializing, defaults to utf-8
    Returns:
        number of bytes written
    Raises:
        AnyMarkupError if a problem occurs while serializing
    """
    try:
        with open(path, 'wb') as f:
            return serialize(struct, format, f, encoding)
    except EnvironmentError as e:
        raise AnyMarkupError(e, traceback.format_exc())