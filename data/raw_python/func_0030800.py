def nullify(v):
    """Convert empty strings and strings with only spaces to None values. """

    if isinstance(v, six.string_types):
        v = v.strip()

    if v is None or v == '':
        return None
    else:
        return v