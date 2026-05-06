def rAsciiLine(ifile):
    """Returns the next non-blank line in an ASCII file."""

    _line = ifile.readline().strip()
    while len(_line) == 0:
        _line = ifile.readline().strip()
    return _line