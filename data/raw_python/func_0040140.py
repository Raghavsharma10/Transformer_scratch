def as_text(str_or_bytes, encoding='utf-8', errors='strict'):
    """Return input string as a text string.

    Should work for input string that's unicode or bytes,
    given proper encoding.

    >>> print(as_text(b'foo'))
    foo
    >>> b'foo'.decode('utf-8') == u'foo'
    True
    """
    if isinstance(str_or_bytes, text):
        return str_or_bytes
    return str_or_bytes.decode(encoding, errors)