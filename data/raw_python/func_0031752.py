def strip_glob(string, split_str=' '):
    """
    Strip glob portion in `string`.

    >>> strip_glob('*glob*like')
    'glob like'
    >>> strip_glob('glob?')
    'glo'
    >>> strip_glob('glob[seq]')
    'glob'
    >>> strip_glob('glob[!seq]')
    'glob'

    :type string: str
    :rtype: str

    """
    string = _GLOB_PORTION_RE.sub(split_str, string)
    return string.strip()