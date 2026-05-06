def displayable_path(path, separator=u'; '):
    """Attempts to decode a bytestring path to a unicode object for the
    purpose of displaying it to the user. If the `path` argument is a
    list or a tuple, the elements are joined with `separator`.
    """
    if isinstance(path, (list, tuple)):
        return separator.join(displayable_path(p) for p in path)
    elif isinstance(path, six.text_type):
        return path
    elif not isinstance(path, bytes):
        # A non-string object: just get its unicode representation.
        return six.text_type(path)

    try:
        return path.decode(_fsencoding(), 'ignore')
    except (UnicodeError, LookupError):
        return path.decode('utf8', 'ignore')