def fsarray(strings, *args, **kwargs):
    """fsarray(list_of_FmtStrs_or_strings, width=None) -> FSArray

    Returns a new FSArray of width of the maximum size of the provided
    strings, or width provided, and height of the number of strings provided.
    If a width is provided, raises a ValueError if any of the strings
    are of length greater than this width"""

    strings = list(strings)
    if 'width' in kwargs:
        width = kwargs['width']
        del kwargs['width']
        if strings and max(len(s) for s in strings) > width:
            raise ValueError("Those strings won't fit for width %d" % width)
    else:
        width = max(len(s) for s in strings) if strings else 0
    fstrings = [s if isinstance(s, FmtStr) else fmtstr(s, *args, **kwargs) for s in strings]
    arr = FSArray(len(fstrings), width, *args, **kwargs)
    rows = [fs.setslice_with_length(0, len(s), s, width) for fs, s in zip(arr.rows, fstrings)]
    arr.rows = rows
    return arr