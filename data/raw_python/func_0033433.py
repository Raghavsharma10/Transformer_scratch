def convert(isbn, code='978'):
    """Convert ISBNs between ISBN-10 and ISBN-13.

    Note:
        No attempt to hyphenate converted ISBNs is made, because the
        specification requires that *any* hyphenation must be correct but
        allows ISBNs without hyphenation.

    Args:
        isbn (str): SBN, ISBN-10 or ISBN-13
        code (str): EAN Bookland code

    Returns:
        ``str``: Converted ISBN-10 or ISBN-13

    Raise:
        IsbnError: When ISBN-13 isn't convertible to an ISBN-10

    """
    isbn = _isbn_cleanse(isbn)
    if len(isbn) == 10:
        isbn = code + isbn[:-1]
        return isbn + calculate_checksum(isbn)
    else:
        if isbn.startswith('978'):
            return isbn[3:-1] + calculate_checksum(isbn[3:-1])
        else:
            raise IsbnError('Only ISBN-13s with 978 Bookland code can be '
                            'converted to ISBN-10.')