def _isbn_cleanse(isbn, checksum=True):
    """Check ISBN is a string, and passes basic sanity checks.

    Args:
        isbn (str): SBN, ISBN-10 or ISBN-13
        checksum (bool): ``True`` if ``isbn`` includes checksum character

    Returns:
        ``str``: ISBN with hyphenation removed, including when called with a
            SBN

    Raises:
        TypeError: ``isbn`` is not a ``str`` type
        IsbnError: Incorrect length for ``isbn``
        IsbnError: Incorrect SBN or ISBN formatting

    """
    if not isinstance(isbn, string_types):
        raise TypeError('ISBN must be a string, received %r' % isbn)

    if PY2 and isinstance(isbn, str):  # pragma: Python 2
        isbn = unicode(isbn)
        uni_input = False
    else:  # pragma: Python 3
        uni_input = True

    for dash in DASHES:
        isbn = isbn.replace(dash, unicode())

    if checksum:
        if not isbn[:-1].isdigit():
            raise IsbnError('non-digit parts')
        if len(isbn) == 9:
            isbn = '0' + isbn
        if len(isbn) == 10:
            if not (isbn[-1].isdigit() or isbn[-1] in 'Xx'):
                raise IsbnError('non-digit or X checksum')
        elif len(isbn) == 13:
            if not isbn[-1].isdigit():
                raise IsbnError('non-digit checksum')
            if not isbn.startswith(('978', '979')):
                raise IsbnError('invalid Bookland region')
        else:
            raise IsbnError('ISBN must be either 10 or 13 characters long')
    else:
        if len(isbn) == 8:
            isbn = '0' + isbn
        elif len(isbn) == 12 and not isbn[:3].startswith(('978', '979')):
            raise IsbnError('invalid Bookland region')
        if not isbn.isdigit():
            raise IsbnError('non-digit parts')
        if not len(isbn) in (9, 12):
            raise IsbnError('ISBN must be either 9 or 12 characters long '
                            'without checksum')
    if PY2 and not uni_input:  # pragma: Python 2
        # Sadly, type ping-pong is required to maintain backwards compatibility
        # with previous pyisbn releases for Python 2 users.
        return str(isbn)
    else:  # pragma: Python 3
        return isbn