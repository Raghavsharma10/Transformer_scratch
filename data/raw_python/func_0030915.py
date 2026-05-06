def parseBtop(btopString):
    """
    Parse a BTOP string.

    The format is described at https://www.ncbi.nlm.nih.gov/books/NBK279682/

    @param btopString: A C{str} BTOP sequence.
    @raise ValueError: If C{btopString} is not valid BTOP.
    @return: A generator that yields a series of integers and 2-tuples of
        letters, as found in the BTOP string C{btopString}.
    """
    isdigit = str.isdigit
    value = None
    queryLetter = None
    for offset, char in enumerate(btopString):
        if isdigit(char):
            if queryLetter is not None:
                raise ValueError(
                    'BTOP string %r has a query letter %r at offset %d with '
                    'no corresponding subject letter' %
                    (btopString, queryLetter, offset - 1))
            value = int(char) if value is None else value * 10 + int(char)
        else:
            if value is not None:
                yield value
                value = None
                queryLetter = char
            else:
                if queryLetter is None:
                    queryLetter = char
                else:
                    if queryLetter == '-' and char == '-':
                        raise ValueError(
                            'BTOP string %r has two consecutive gaps at '
                            'offset %d' % (btopString, offset - 1))
                    elif queryLetter == char:
                        raise ValueError(
                            'BTOP string %r has two consecutive identical %r '
                            'letters at offset %d' %
                            (btopString, char, offset - 1))
                    yield (queryLetter, char)
                    queryLetter = None

    if value is not None:
        yield value
    elif queryLetter is not None:
        raise ValueError(
            'BTOP string %r has a trailing query letter %r with '
            'no corresponding subject letter' % (btopString, queryLetter))