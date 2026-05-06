def as_string(value):
    """Convert a value to a Unicode object for matching with a query.
    None becomes the empty string. Bytestrings are silently decoded.
    """
    if six.PY2:
        buffer_types = buffer, memoryview  # noqa: F821
    else:
        buffer_types = memoryview

    if value is None:
        return u''
    elif isinstance(value, buffer_types):
        return bytes(value).decode('utf8', 'ignore')
    elif isinstance(value, bytes):
        return value.decode('utf8', 'ignore')
    else:
        return six.text_type(value)