def lineAndColumnAt(s, pos):
    r"""Return line and column of `pos` (0-based!) in `s`. Lines start with
    1, columns with 0.

    Examples:

    >>> lineAndColumnAt("0123\n56", 5)
    (2, 0)
    >>> lineAndColumnAt("0123\n56", 6)
    (2, 1)
    >>> lineAndColumnAt("0123\n56", 0)
    (1, 0)
    """
    if pos >= len(s):
        raise IndexError("`pos` %d not in string" % pos)
    # *don't* count last '\n', if it is at pos!
    line = s.count('\n',0,pos)
    if line:
        return line + 1, pos - s.rfind('\n',0,pos) - 1
    else:
        return 1, pos