def to_unicode(text):
    """ Return a decoded unicode string.
        False values are returned untouched.
    """
    if not text or isinstance(text, unicode if PY2 else str):
        return text

    try:
        # Try UTF-8 first
        return text.decode("UTF-8")
    except UnicodeError:
        try:
            # Then Windows Latin-1
            return text.decode("CP1252")
        except UnicodeError:
            # Give up, return byte string in the hope things work out
            return text