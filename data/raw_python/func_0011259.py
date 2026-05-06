def description_of(lines, name='stdin'):
    """
    Return a string describing the probable encoding of a file or
    list of strings.

    :param lines: The lines to get the encoding of.
    :type lines: Iterable of bytes
    :param name: Name of file or collection of lines
    :type name: str
    """
    u = UniversalDetector()
    for line in lines:
        u.feed(line)
    u.close()
    result = u.result
    if result['encoding']:
        return '{0}: {1} with confidence {2}'.format(name, result['encoding'],
                                                     result['confidence'])
    else:
        return '{0}: no result'.format(name)