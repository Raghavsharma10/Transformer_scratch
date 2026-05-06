def chars2str(chars) -> List[str]:
    """Inversion function of function |str2chars|.

    >>> from hydpy.core.netcdftools import chars2str

    >>> chars2str([[b'z', b'e', b'r', b'o', b's'],
    ...            [b'o', b'n', b'e', b's', b'']])
    ['zeros', 'ones']

    >>> chars2str([])
    []
    """
    strings = collections.deque()
    for subchars in chars:
        substrings = collections.deque()
        for char in subchars:
            if char:
                substrings.append(char.decode('utf-8'))
            else:
                substrings.append('')
        strings.append(''.join(substrings))
    return list(strings)