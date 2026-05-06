def slice(string, start=None, end=None):
    """
    Returns a substring of the given string, counting graphemes instead of codepoints.

    Negative indices is currently not supported.
    >>> string = "tamil நி (ni)"

    >>> string[:7]
    'tamil ந'
    >>> grapheme.slice(string, end=7)
    'tamil நி'
    >>> string[7:]
    'ி (ni)'
    >>> grapheme.slice(string, 7)
    ' (ni)'
    """

    if start is None:
        start = 0
    if end is not None and start >= end:
        return ""

    if start < 0:
        raise NotImplementedError("Negative indexing is currently not supported.")

    sum_ = 0
    start_index = None
    for grapheme_index, grapheme_length in enumerate(grapheme_lengths(string)):
        if grapheme_index == start:
            start_index = sum_
        elif grapheme_index == end:
            return string[start_index:sum_]
        sum_ += grapheme_length

    if start_index is not None:
        return string[start_index:]

    return ""