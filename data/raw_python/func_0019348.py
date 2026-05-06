def str2chars(strings) -> numpy.ndarray:
    """Return |numpy.ndarray| containing the byte characters (second axis)
    of all given strings (first axis).

    >>> from hydpy.core.netcdftools import str2chars
    >>> str2chars(['zeros', 'ones'])
    array([[b'z', b'e', b'r', b'o', b's'],
           [b'o', b'n', b'e', b's', b'']],
          dtype='|S1')

    >>> str2chars([])
    array([], shape=(0, 0),
          dtype='|S1')
    """
    maxlen = 0
    for name in strings:
        maxlen = max(maxlen, len(name))
    # noinspection PyTypeChecker
    chars = numpy.full(
        (len(strings), maxlen), b'', dtype='|S1')
    for idx, name in enumerate(strings):
        for jdx, char in enumerate(name):
            chars[idx, jdx] = char.encode('utf-8')
    return chars