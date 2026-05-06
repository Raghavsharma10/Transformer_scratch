def ai(board, who='x'):
    """
    Returns best next board

    >>> b = Board(); b._rows = [['x', 'o', ' '], ['x', 'o', ' '], [' ', ' ', ' ']]
    >>> ai(b)
    < Board |xo.xo.x..| >
    """
    return sorted(board.possible(), key=lambda b: value(b, who))[-1]