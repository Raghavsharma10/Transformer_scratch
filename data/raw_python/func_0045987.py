def text_coords(string, position):
    r"""
    Transform a simple index into a human-readable position in a string.

    This function accepts a string and an index, and will return a triple of
    `(lineno, columnno, line)` representing the position through the text. It's
    useful for displaying a string index in a human-readable way::

        >>> s = "abcdef\nghijkl\nmnopqr\nstuvwx\nyz"
        >>> text_coords(s, 0)
        (0, 0, 'abcdef')
        >>> text_coords(s, 4)
        (0, 4, 'abcdef')
        >>> text_coords(s, 6)
        (0, 6, 'abcdef')
        >>> text_coords(s, 7)
        (1, 0, 'ghijkl')
        >>> text_coords(s, 11)
        (1, 4, 'ghijkl')
        >>> text_coords(s, 15)
        (2, 1, 'mnopqr')
    """
    line_start = string.rfind('\n', 0, position) + 1
    line_end = string.find('\n', position)
    lineno = string.count('\n', 0, position)
    columnno = position - line_start
    line = string[line_start:line_end]
    return (lineno, columnno, line)